# Load module for animation of loading
from tqdm import tqdm

import cv2
import numpy as np
from gaussian import find_gaussian_parameters

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
for i in tqdm(range(int(len(np_frame))), desc="Processing data"):
    # Process each frame
    data = []
    class_probabilities, mus, sigmas = find_gaussian_parameters(data, K, n_iterations)
    print(class_probabilities)
    print(mus)
    print(sigmas)
    pass

# Closing the VideoCapture object
read_video.release()

# Destroy all the open cv2 windows
cv2.destroyAllWindows()
