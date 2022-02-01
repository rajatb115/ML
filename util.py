import configparser
import cv2 as cv
import numpy as np

debug = False


def read_config(config_path):
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Debug
    if debug:
        log_info = open("data/log.txt", 'a')
        log_info.write("# At util values are : \n")
        log_info.write("Input : " + str(parser['config']['input']) + "\n")
        log_info.write("Output : " + str(parser['config']['output']) + "\n")
        log_info.write("Alpha : " + str(parser['config']['alpha']) + "\n")
        log_info.write("Weight : " + str(parser['config']['weight']) + "\n")
        log_info.write("Variance : " + str(parser['config']['variance']) + "\n")
        log_info.write("Number of gaussian : " + str(parser['config']['number_gaussian']) + "\n")
        log_info.write("Weight threshold : " + str(parser['config']['weight_threshold']) + "\n")
        log_info.write("\n")
        log_info.close()

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
        log_info = open("data/log.txt", 'a')
        log_info.write("# At util values are :\n")
        log_info.write("Number of frames : " + str(cnt) + "\n")
        log_info.write("\n")
        log_info.close()
    return cnt


def add_frame_foreground(frame, write_pointer, height, width):
    # Adding the current frame to the video
    # convert this frame into frame format

    # temp_frame = np.concatenate(frame, axis=0)
    # new_frame = temp_frame.reshape((height, width, 3))

    write_pointer.write(frame)


def add_frame_background(foreground_frame, frame, write_pointer1, height, width):
    write_pointer1.write()
