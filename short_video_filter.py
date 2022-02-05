import os
import cv2
import glob
import numpy as np
from shutil import copy
import ffmpeg
import imutils
import argparse
import json

from wechat_utils import choose_whether_to_use_existing_output_directory, get_wechat_dirs
from settings import *

verbose = False
show = False


def extract_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_WHITE_RANGE[0], HSV_WHITE_RANGE[1])
    extracted = cv2.bitwise_and(img, img, mask=mask)
    return mask, extracted


def show_match_result(img, loc, shape, reverseScale, color=(0, 0, 255)):
    cv2.rectangle(img, (int(loc[0]*reverseScale), int(loc[1]*reverseScale)),
                  (int((loc[0] + shape[1])*reverseScale), int((loc[1] + shape[0])*reverseScale)), color, 2)
    cv2.imshow('match result', img)
    cv2.waitKey(0)


def generate_scales(template, image):
    scales = []
    minScale = max(template.shape[0]/image.shape[0],
                   template.shape[1]/image.shape[1])
    maxScale = min(SCALE_RANGE*minScale, SCALE_RANGE)
    increase = (maxScale - minScale) / (SCALE_NUM - 1)
    scale = minScale
    for i in range(0, SCALE_NUM):
        scales.append(scale)
        scale += increase
    return scales


def multi_scale_template_match(template, image, scales=[1, ]):
    templateImg = cv2.cvtColor(template.image, cv2.COLOR_BGR2GRAY)
    templateEdge = cv2.Canny(templateImg, 50, 200)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    for scale in scales:
        result = scale_template_match(templateEdge, imageGray, scale)
        if(show):
            show_match_result(
                image, result[1], template.image.shape, result[2], template.color)
        if found is None or result[0] > found[0]:
            found = result

    if(verbose):
        print("found values:", found)
    return found[0] > MATCH_THRESHOLD


def scale_template_match(templateEdge, imageGray, scale):
    resized = imutils.resize(
        imageGray, width=int(imageGray.shape[1] * scale))
    r = imageGray.shape[1] / float(resized.shape[1])
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, templateEdge, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    return (maxVal, maxLoc, r)


class Video:
    def __init__(self, filePath):
        self.filePath = filePath
        self.vidcap = self.reset_vidcap()
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.frameNum = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frameIdx = 0
        self.frame = None
        self.frame_list = []
        self.frame_list_idx = 0
        self.frame_list_len = 0
        self.frame_list_max_len = SAMPLE_NUM
        self.metadata = self.extract_metadata()
        self.pixel_areas = []

    def reset_vidcap(self):
        self.vidcap = cv2.VideoCapture(self.filePath)
        return self.vidcap

    def is_rotated(self):
        if('side_data_list' in self.metadata):
            for data in self.metadata['side_data_list']:
                if('rotation' in data):
                    return True
            return False
        else:
            return False

    def extract_metadata(self):
        metadata = None
        vid = ffmpeg.probe(self.filePath)
        for i in range(len(vid['streams'])):
            if(vid['streams'][i]['codec_type'] == 'video'):
                metadata = vid['streams'][i]
        return metadata

    def is_short_video_format(self):
        return self.metadata["coded_height"] / self.metadata["coded_width"] >= SHORT_VIDEO_HEIGHT_WIDTH_RATIO and not self.is_rotated()

    def get_cut_area(self):
        if(self.pixel_areas == []):
            for area in SELECTED_AREAS:
                self.pixel_areas.append([int(area[0] * self.height), int(area[1] *
                                                                         self.height), int(area[2] * self.width), int(area[3] * self.width)])
        return self.pixel_areas


class TemplateImage:
    def __init__(self, filePath):
        self.filePath = filePath
        self.name = os.path.basename(filePath)
        self.image = cv2.imread(filePath)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.color = (0, 0, 255)

    @classmethod
    def get_all_templates(cls):
        fjson = open(os.path.join(RESOURCE_DIR, "meta.json"))
        data = json.load(fjson)
        fjson.close()
        templates = []
        for file in os.listdir(RESOURCE_DIR):
            if(file.endswith('.jpg')):
                templates.append(TemplateImage(
                    os.path.join(RESOURCE_DIR, file)))
                templates[-1].color = data[file]['color']
        return templates


class VideoClassifier:
    def __init__(self):
        self.templates = TemplateImage.get_all_templates()

    def classify_image_partially(self, img, scales=[1, ], extractWhite=False):
        areas = self.currentVideo.get_cut_area()
        for area in areas:
            kind = self.classify_image(
                img[area[0]:area[1], area[2]:area[3]], scales, extractWhite)
            if(kind is not None):
                return kind

    def classify_image(self, img, scales=[1, ], extractWhite=False):
        # imageData = np.dstack(img)
        if(extractWhite):
            _, img = extract_white(img)
        if(show):
            cv2.imshow('image', img)
            cv2.waitKey(0)
        kind = None
        for template in self.templates:
            isMatch = multi_scale_template_match(
                template, img, scales)
            if(isMatch):
                kind = template.name
            if(verbose):
                print(template.name, "isMatch:", isMatch)
        return kind

    def classify_by_diff(self, video):
        vidcap = video.reset_vidcap()
        # nb_frames = int(metadata['nb_frames'])
        r_frame_rate = video.metadata['r_frame_rate'].split('/')
        fps = int(r_frame_rate[0]) / int(r_frame_rate[1])
        frame_interval = int(DIFF_INTERVAL * fps)
        finishFrame = video.frameNum - \
            int(CUT_TIME_LENGTH * fps)
        current = 0
        haveLastImg = False
        lastImg = None
        kind = None
        diffSum = None
        diffCnt = 0
        while(current < finishFrame):
            success, img = vidcap.read()
            if(not success):
                break
            current += 1
            if(current % frame_interval == 0):
                if(show and haveLastImg):
                    diff = cv2.absdiff(lastImg, img)
                    diffGray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    if(diffSum is None):
                        diffSum = diffGray
                        diffCnt += 1
                    else:
                        diffSum += diffGray
                        diffCnt += 1
                    # if(show):
                    #     cv2.imshow('image', diffGray)
                    #     cv2.waitKey(0)
                lastImg = img
                haveLastImg = True
        if(diffCnt > 0):
            diffSum = diffSum // diffCnt
            if(show):
                print("show diff")
                cv2.imshow('image', diffSum)
                cv2.waitKey(0)
        return kind

    def classify_last_frame(self, img, originScales):
        if(verbose):
            print("classify by last frame")
            cv2.imwrite("lab/files/last_frame.jpg".format(), img)
        if(show):
            cv2.imshow('image', img)
            cv2.waitKey(0)
        kind = None
        for template in self.templates:
            scales = generate_scales(template.image, img)
            scales += originScales
            isMatch = multi_scale_template_match(
                template, img, scales)
            if(isMatch):
                kind = template.name
            if(verbose):
                print(template.name, "isMatch:", isMatch)
        return kind

    def classify_by_match(self, video, extractWhite=False):
        vidcap = video.reset_vidcap()
        kind = None
        interval = video.frameNum // (SAMPLE_NUM - 1)
        currentIndex = 0
        scale = [720 / video.width, ]
        suc, img = None, None
        for j in range(SAMPLE_NUM - 1):
            suc, img = vidcap.read()
            if(not suc):
                break
            res = self.classify_image_partially(img, scale, extractWhite)
            if(res):
                kind = res
                return kind
            for i in range(interval - 1):
                suc, img = vidcap.read()
            currentIndex += interval
        while(suc):
            suc, newImg = vidcap.read()
            if(suc):
                img = newImg
        kind = self.classify_last_frame(img, scale)
        return kind

    def classify_video(self, video):
        self.currentVideo = video
        if(not video.is_short_video_format()):
            return None
        kind = self.classify_by_match(video, True)
        if(verbose):
            print("match classify result:", kind)
        if(kind):
            return kind
        kind = self.classify_by_diff(video)
        if(verbose):
            print("diff classify result:", kind)
        if(kind):
            return kind
        return 'short_video'


def move_file(filePath, kind, description):
    if(kind):
        kindPath = os.path.join(args.output, kind)
        if(not os.path.exists(kindPath)):
            os.mkdir(kindPath)
        copy(filePath, kindPath + '/' + description + '.mp4')
    else:
        kindPath = os.path.join(args.output, 'unclassified')
        if(not os.path.exists(kindPath)):
            os.mkdir(kindPath)
        copy(filePath, kindPath + '/' + description + '.mp4')


def filter_short_videos(args):
    if(os.path.exists(args.output)):
        choose_whether_to_use_existing_output_directory(args.output)
    else:
        os.mkdir(args.output)

    dirs = get_wechat_dirs(args.input)
    classifier = VideoClassifier()
    for directory in dirs:
        if(verbose):
            print("handling", directory)
        dirPath = os.path.join(args.input, directory)
        files = os.listdir(dirPath)
        cnt = 0
        for file in files:
            if(file.endswith('.mp4')):
                filePath = os.path.join(dirPath, file)
                if(verbose):
                    print("handling", filePath)
                kind = classifier.classify_video(Video(filePath))
                description = directory + '_' + str(cnt) + '_' + file[:-4]
                move_file(filePath, kind, description)
                cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '-i', '--input', help='Input video directory', default='.')
    parser.add_argument(
        '-o', '--output', help='Output video directory', default=None)
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Add verbose log'
    )
    parser.add_argument(
        '-s', '--show', action='store_true', help='Show images'
    )
    args = parser.parse_args()
    print(args)
    if(not args.output):
        args.output = os.path.join(args.input, 'arranged')
    verbose = args.verbose
    show = args.show
    if(show):
        cv2.namedWindow('image', 0)
        cv2.namedWindow('match result', 1)

    filter_short_videos(args)
