import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Brute-Force Matching with ORB Descriptors

img1 = cv.imread('../SIFT_SURF/FeatureMatching/images/box.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('../SIFT_SURF/FeatureMatching/images/box_in_scene.png',cv.IMREAD_GRAYSCALE) # trainImage

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object with Hamming distance as measurement and crossCheck is True for better results
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.to get the best matches in two images
matches = bf.match(des1,des2)

# Sort them in the order of their distance,so that best matches(with low distance) come to front.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()

