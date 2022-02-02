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


def find_background(frame, tmp_frame_foreground, max_history, frame_history, frame_history_cnt, width, height):
    new_frame = np.zeros((height, width, 3))

    if frame_history_cnt <= max_history:
        for i in range(height):
            for j in range(width):
                if (tmp_frame_foreground[i, j, 0] != 255 or tmp_frame_foreground[i, j, 1] != 255 or
                        tmp_frame_foreground[i, j, 0] != 255):
                    new_frame[i, j, 0] = np.mean(np.array(frame_history)[:, i, j, 0])
                    new_frame[i, j, 1] = np.mean(np.array(frame_history)[:, i, j, 1])
                    new_frame[i, j, 2] = np.mean(np.array(frame_history)[:, i, j, 2])

                else:
                    new_frame[i, j, 0] = frame[i, j, 0]
                    new_frame[i, j, 1] = frame[i, j, 1]
                    new_frame[i, j, 2] = frame[i, j, 2]
        frame_history_cnt += 1
        frame_history.append(new_frame)
    else:
        frame_history = frame_history[1:]
        for i in range(height):
            for j in range(width):
                if (tmp_frame_foreground[i, j, 0] != 255 or tmp_frame_foreground[i, j, 1] != 255 or
                        tmp_frame_foreground[i, j, 0] != 255):
                    new_frame[i, j, 0] = np.mean(np.array(frame_history)[:, i, j, 0])
                    new_frame[i, j, 1] = np.mean(np.array(frame_history)[:, i, j, 1])
                    new_frame[i, j, 2] = np.mean(np.array(frame_history)[:, i, j, 2])

                else:
                    new_frame[i, j, 0] = frame[i, j, 0]
                    new_frame[i, j, 1] = frame[i, j, 1]
                    new_frame[i, j, 2] = frame[i, j, 2]
        frame_history.append(new_frame)
    return new_frame, frame_history, frame_history_cnt
