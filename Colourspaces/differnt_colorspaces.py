import cv2

# It will get loaded in BGR format by default.

# The first image is taken under outdoor conditions with bright sunlight
bright = cv2.imread('../COLOURSPACES/cube1.jpg')
B, G, R = cv2.split(bright)

# Corresponding channels are separated
cv2.imshow("original_bright", bright)
cv2.waitKey(0)

cv2.imshow("blue", B)
cv2.waitKey(0)

cv2.imshow("Green", G)
cv2.waitKey(0)

cv2.imshow("red", R)
cv2.waitKey(0)

# the second is taken indoor with normal lighting conditions.
dark = cv2.imread('../COLOURSPACES/cube2.jpg')
B, G, R = cv2.split(dark)

cv2.imshow("original_dark", dark)
cv2.waitKey(0)

cv2.imshow("blue", B)
cv2.waitKey(0)

cv2.imshow("Green", G)
cv2.waitKey(0)

cv2.imshow("red", R)
cv2.waitKey(0)

# The LAB Color-Space

brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
cv2.imshow("original_brightLAB", brightLAB)
cv2.waitKey(0)

darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
cv2.imshow("original_darkLAB", darkLAB)
cv2.waitKey(0)

#The YCrCb Color-Space

brightYCB = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
cv2.imshow("original_brightYCB", brightYCB)
cv2.waitKey(0)

darkYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)
cv2.imshow("original_darkYCB", darkYCB)
cv2.waitKey(0)

# The HSV Color Space

brightHSV = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
cv2.imshow("original_brightHSV", brightHSV)
cv2.waitKey(0)

darkHSV = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)
cv2.imshow("original_darkHSV", darkHSV)
cv2.waitKey(0)

cv2.destroyAllWindows()