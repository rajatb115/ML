import configparser
import cv2 as cv
import numpy as np

debug = True


def read_config(config_path):
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Debug
    if debug:
        print("# At util values are :")
        print("Input :", parser['config']['input'])
        print("Output :", parser['config']['output'])
        print()

    return parser


def find_frames(input_path):
    # Read video from the disk
    read_video = cv.VideoCapture(input_path)

    # Check if video is open to read properly
    if not read_video.isOpened():
        print("# Error while loading the video. Please check :(")
        exit()
    cnt = 0
    while read_video.isOpened():
        # ret : is a boolean variable that returns true if the frame is available. frame : is an image array vector
        # captured based on the default frames per second defined explicitly or implicitly
        ret, frame = read_video.read()
        if ret:
            cnt += 1
        else:
            break
    read_video.release()

    if debug:
        print("# At util values are :")
        print("Number of frames :", cnt)
        print()
    return cnt


def add_frame(frame, write_pointer, height, width):
    # Adding the current frame to the video
    # convert this frame into frame format

    temp_frame = np.concatenate(frame, axis=0)
    new_frame = temp_frame.reshape((height, width, 3))
    write_pointer.write(new_frame)
