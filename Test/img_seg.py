# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be thresholded")
ap.add_argument("-t", "--threshold", type = int, default = 128,
	help = "Threshold value")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize the list of threshold methods

methods = [
	("BINARY", cv2.THRESH_BINARY),
	("BINARY_INV", cv2.THRESH_BINARY_INV),
	("TRUNC", cv2.THRESH_TRUNC),
	("TOZERO", cv2.THRESH_TOZERO),
	("TOZERO_INV", cv2.THRESH_TOZERO_INV)]
 

# show threshold methods Binary will be the best
for (threshName, threshMethod) in methods:
	# threshold the image and show it
	(T, thresh) = cv2.threshold(gray, args["threshold"], 255, threshMethod)
	cv2.imshow(threshName, thresh)
	cv2.waitKey(0)


