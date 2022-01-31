import sys
import time
from multiprocessing import Pool

import numpy as np

import gaussian
import util
import cv2 as cv
from tqdm import tqdm

debug = True

if __name__ == "__main__":
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
    write_video = cv.VideoWriter(str(config['config']['output']), video_compression, fps, (width, height))

    # Make few variables to store (i.e. store the history)
    # Mean of each gaussian per pixel (RGB)
    # Weights of each gaussian per pixel
    # Variance of each gaussian per pixel
    # Number of Gaussian
    mean_np = np.zeros((height * width, 3*int(config['config']['number_gaussian'])))
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

        new_frame = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                new_frame[i, j] = gaussian.process_pixel([reshaped_frame[i*height+j], mean_np[i*height+j], weight_np[i*height+j], var_np[i*height+j], number_gaussian_np[i*height+j], config])

        # add the frame to the video
        util.add_frame(new_frame, write_video, height, width)

    end_time = time.time()
    print("Total time taken by the algorithm to process the video is :", end_time-start_time, "seconds")

    read_video.release()
    write_video.release()
    cv.destroyAllWindows()
