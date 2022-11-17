import argparse
import imutils
import cv2

ia = [1, 2, 3]
ie = [3, 4, 5]

ia = ie + ia

# construct the argument parser and parse the arguments
image = cv2.imread('D:/GitHub/Computer_vision/new/whiteballs.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', image)
cv2.imshow('gray', gray)

# blur the image (to reduce false-positive detections) and then
# perform edge detection
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(blurred, 50, 130)
# cv2.imshow("blur",blurred)
cv2.imshow('edged', edged)

# find contours in the edge map and initialize the total number of
# shapes found
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
total = 0

# loop over the contours one by one
for c in cnts:
    # if the contour area is small, then the area is likely noise, so
    # we should ignore the contour
    if cv2.contourArea(c) < 25:
        continue

    # otherwise, draw the contour on the image and increment the total
    # number of shapes found
    cv2.drawContours(image, [c], -1, (204, 0, 255), 2)
    total += 1

# show the output image and the final shape count
print("[INFO] found {} shapes".format(total))
cv2.imshow("Image", image)
cv2.waitKey(0)