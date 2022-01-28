# Load module for animation of loading
from tqdm import tqdm

import cv2
import numpy as np
from gaussian import find_gaussian_parameters

def process_data(i,t,np_frame):
    data_frames = np_frame[max(0, i-t): i]
    frame_shape = data_frames[0].shape

    frame_pixel = []
    for row in range(frame_shape[0]):
        tmp = []
        for col in range(frame_shape[1]):
            tmp.append([])

        frame_pixel.append(tmp)

    #print(len(frame_pixel[0]))
    for _ in range(len(data_frames)):
        for row in range(frame_shape[0]):
            for col in range(frame_shape[1]):
                frame_pixel[row][col].append(data_frames[_][row][col])

    return frame_pixel
# Reading video
read_video = cv2.VideoCapture('data/input_video.mpg')

# Check if video is open to read properly
if not read_video.isOpened():
    print("# Error while loading the video. Please check :(")
    exit()

np_frame = []

# If video is open to read properly then read it frame by frame
while read_video.isOpened():

    # ret : is a boolean variable that returns true if the frame is available.
    # frame : is an image array vector captured based on the default frames per second defined explicitly or implicitly
    ret, frame = read_video.read()

    if ret:
        cv2.imwrite('data/video_frame.jpg', frame)
        img = cv2.imread('data/video_frame.jpg')

        np_frame.append(img)
    else:
        break

print("# Number of frames in the video = ", len(np_frame))

# Processing each frame
# Adding animation for loading
K = 3
n_iterations = 10
t = 10

for i in tqdm(range(4,int(len(np_frame))), desc="Processing data"):
    # Process each frame for time t

    data_lst = process_data(i+1, t, np_frame)

    for __ in range(len(data_lst)):
        for _ in range(len(data_lst[0])):
            final_lst = []
            for j in range(min(t, i+1)):
                final_lst.append(data_lst[__][_][j][0])

            data = np.array(final_lst)

            class_probabilities, mus, sigmas = find_gaussian_parameters(data, K, n_iterations)
            print(class_probabilities)
            print(mus)
            print(sigmas)
    break

# Closing the VideoCapture object
read_video.release()

# Destroy all the open cv2 windows
cv2.destroyAllWindows()
