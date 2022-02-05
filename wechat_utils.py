import os
import re


def get_wechat_dirs(inputPath):
    months = os.listdir(inputPath)
    pattern = re.compile(r"[0-9]*-[0-9]*")
    dirs = []
    for month in months:
        if(pattern.match(month)):
            dirs.append(month)
    return dirs


def choose_whether_to_use_existing_output_directory(dirPath):
    while(True):
        print('The output directory', dirPath,
              'already exists. Do you want to use it? (y/n)')
        choice = input()
        if(choice.lower() == 'y'):
            return True
        elif(choice.lower() == 'n'):
            return False


def check_and_make_directory(dirPath):
    if(os.path.exists(dirPath)):
        return choose_whether_to_use_existing_output_directory(dirPath)
    else:
        os.mkdir(dirPath)
        return True
