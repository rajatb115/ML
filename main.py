import sys
import time
from multiprocessing import Pool

import numpy as np

import gaussian
import util
import cv2 as cv
from tqdm import tqdm

debug = False

if __name__ == "__main__":

    # cleaning the log file
    if debug:
        log_info = open("data/log.txt", 'w')
        log_info.write("")
        log_info.close()

    config_path = sys.argv[1]
    config = util.read_config(config_path)

    # Read video from the disk
    read_video = cv.VideoCapture(config['config']['input'])

    # Check if video is open to read properly
    if not read_video.isOpened():
        print("# Error while loading the video. Please check :(")
        exit()

    # Reading width, height and fps from the input video
    width = int(read_video.get(3))
    height = int(read_video.get(4))
    fps = int(read_video.get(5))
    frame_count = util.find_frames(config['config']['input'])

    # Debug the code
    if debug:
        print("# At main values are :")
        print("Width of the video :", width)
        print("Height of the video :", height)
        print("Frame per second of the video :", fps)
        print("Total frames in the video :", frame_count)
        print()

    # Write video to the disk
    # Selecting the video compression
    video_compression = cv.VideoWriter_fourcc(*'XVID')
    video_compression1 = cv.VideoWriter_fourcc(*'XVID')
    write_video_background = cv.VideoWriter(str(config['config']['output_background']), video_compression, fps, (width, height))
    write_video_foreground = cv.VideoWriter(str(config['config']['output_foreground']), video_compression1, fps, (width, height))

    # Make few variables to store (i.e. store the history)
    # Mean of each gaussian per pixel (RGB)
    # Weights of each gaussian per pixel
    # Variance of each gaussian per pixel
    # Number of Gaussian
    mean_np = np.zeros((height * width, 3 * int(config['config']['number_gaussian'])))
    weight_np = np.zeros((height * width, int(config['config']['number_gaussian'])))
    var_np = np.zeros((height * width, int(config['config']['number_gaussian'])))
    number_gaussian_np = np.zeros((height * width))

    # Read each frame and fit the gaussian and update the values
    start_time = time.time()

    for _ in tqdm(range(frame_count), desc="Processing data"):

        # Reading the frame. frame is of the format height * width * 3 (RGB)
        ret, frame = read_video.read()

        # Reformatting the frame matrix
        reshaped_frame = frame.reshape((width * height, 3))
        new_frame_foreground = np.zeros((width * height, 3))
        new_frame_background = np.zeros((width * height, 3))

        data_prep = []
        for i in range(height):
            for j in range(width):
                # new_frame[i * width + j], mean_np[i * width + j], var_np[i * width + j], weight_np[i * width + j],
                # number_gaussian_np[i*width + j] = gaussian.process_pixel([reshaped_frame[i * width + j],
                # mean_np[i * width + j], weight_np[i * width + j],var_np[i * width + j], number_gaussian_np[i *
                # width + j], config])
                data_prep.append([reshaped_frame[i * width + j], mean_np[i * width + j], weight_np[i * width + j], var_np[i * width + j], number_gaussian_np[i * width + j], config])

        process = Pool(processes=int(config['config']['num_process']))
        val = process.map(gaussian.process_pixel, data_prep)
        process.close()

        for i in range(height):
            for j in range(width):
                new_frame_background[i * width + j], new_frame_foreground[i*width+j], mean_np[i * width + j], var_np[i * width + j], weight_np[i * width + j], number_gaussian_np[i * width + j] = val[i*width+j]

        tmp_frame_background = np.reshape(np.concatenate(new_frame_background, axis=0), (height, width, 3))
        tmp_frame_foreground = np.reshape(np.concatenate(new_frame_foreground, axis=0), (height, width, 3))

        # add the frame to the video
        tmp_new_frame_foreground = np.uint8(tmp_frame_foreground)
        tmp_new_frame_background = np.uint8(tmp_frame_background)

        cv.imwrite('data/video_frame_back.jpg', tmp_new_frame_background)
        cv.imwrite('data/video_frame_fore.jpg', tmp_new_frame_foreground)

        #util.add_frame_foreground(tmp_new_frame, write_video_foreground, height, width)
        write_video_background.write(tmp_new_frame_background)
        write_video_foreground.write(tmp_new_frame_foreground)

        if _ == 25:
            break
    end_time = time.time()
    print("Total time taken by the algorithm to process the video is :", end_time - start_time, "seconds")

    read_video.release()
    write_video_foreground.release()
    write_video_background.release()
    cv.destroyAllWindows()
