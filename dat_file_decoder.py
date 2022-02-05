import os
import argparse
from shutil import copy
from wechat_utils import check_and_make_directory, choose_whether_to_use_existing_output_directory, get_wechat_dirs

# Image formats supported by WeChat
# JPEG: 0xFFD8FF
# PNG: 0x89504E47
# GIF: 0x47494638
# BMP: 0x424D
formats = [
    {
        'name': 'JPEG',
        'value': b'\xFF\xD8\xFF',
        'extension': 'jpg'
    },
    {
        'name': 'PNG',
        'value': b'\x89\x50\x4E\x47',
        'extension': 'png'
    },
    {
        'name': 'GIF',
        'value': b'\x47\x49\x46\x38',
        'extension': 'gif'
    },
    {
        'name': 'BMP',
        'value': b'\x42\x4D',
        'extension': 'bmp'
    },
]

headerMap = {}
key = None
HEADER_LENGTH = 4


def init():
    for i in range(len(formats)):
        print(formats[i]['value'][:2])
        headerMap[formats[i]['value'][:2]] = i


def get_xor_key(filePath):
    if(key):
        return key
    f = open(filePath, 'rb')
    header = f.read(4)
    for format in formats:
        last_result = None
        foundKey = True
        for i in range(len(format['value'])):
            print(format['value'][i])
            result = header[i] ^ format['value'][i]
            if(i != 0):
                if(last_result != result):
                    foundKey = False
                    break
            last_result = result
        if(foundKey):
            return result
    raise Exception('No key found')


def classify_format(bytes):
    print(bytes)
    decodedBytes = b''
    print(key, bytes[0])
    for i in range(len(bytes)):
        decodedBytes += (bytes[i] ^ key).to_bytes(1, 'big')
    try:
        index = headerMap[decodedBytes[:2]]
    except:
        if(bytes[:2] in headerMap):
            print('The file is not decoded, maybe this is a damaged image')
            index = headerMap[bytes[:2]]
            decodedBytes = bytes
        else:
            # Special files will not be encoded by WeChat
            raise Exception('Unknown format with header',
                            bytes[:2], 'or', decodedBytes[:2])
    if(decodedBytes[:len(formats[index]['value'])] == formats[index]['value']):
        return index
    else:
        raise Exception('Unknown format')


def decode_file(dirPath, fileName, outputDirPath):
    filePath = dirPath + '/' + fileName
    fdat = open(filePath, 'rb')
    header = fdat.read(HEADER_LENGTH)
    print(fileName)
    try:
        formatIndex = classify_format(header)
    except:
        copy(filePath, outputDirPath + '/' + fileName)
        return
    outputPath = outputDirPath + '/' + \
        fileName[:-4] + '.' + formats[formatIndex]['extension']
    fout = open(outputPath, 'wb')
    for i in range(HEADER_LENGTH):
        fout.write((header[i] ^ key).to_bytes(1, 'big'))
    bytes = fdat.read()
    for byte in bytes:
        fout.write((byte ^ key).to_bytes(1, 'big'))


def decode_all_files_in_dir(dirPath, outputDirPath):
    global key
    files = os.listdir(dirPath)
    if(not check_and_make_directory(outputDirPath)):
        return
    if(key == None):
        key = get_xor_key(dirPath + '/' + files[0])
    for file in files:
        if(file.endswith('.dat')):
            decode_file(dirPath, file, outputDirPath)


def decode_all_dirs(dirPath, outputDirPath):
    dirs = get_wechat_dirs(dirPath)
    if(not check_and_make_directory(outputDirPath)):
        return
    for dir in dirs:
        decode_all_files_in_dir(dirPath + '/' + dir, outputDirPath + '/' + dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-d', '--dir', help='Input directory')
    parser.add_argument('-s', '--single', action='store_true',
                        help='Single directory to decode')
    parser.add_argument('-f', '--file', help='File to decode')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-k', '--key', help='Key to decode by xor')
    args = parser.parse_args()
    init()
    if args.dir:
        if(args.single):
            decode_all_files_in_dir(args.dir, args.output)
        else:
            decode_all_dirs(args.dir, args.output)
    elif args.file:

        decode_file(os.path.dirname(args.file),
                    os.path.basename(args.file), args.output)
    else:
        print('No arguments given')
