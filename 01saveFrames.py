
import cv2

import math
import numpy as np

dir = "trainData/"

import getopt,sys

if __name__=="__main__": 
	cv2.namedWindow( "WebCAM 0", 1 ) 

	capture = cv2.VideoCapture(0)
	capture.set(3,320)
	capture.set(4,240)

# max 10 objects
	i = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	c = 0
	t = 0
	saveFrame = 0
	while True: 
		ret, img = capture.read()
		cv2.imshow("WebCAM 0", img) 
		if saveFrame == 1:
			cv2.imwrite(dir+"object"+str(t)+"-"+str(i[t]).zfill(5)+".png",img)
			print "object"+str(t)+"-"+str(i[t]).zfill(5)+".png"
			i[t] += 1
			c += 1

		if c == 1:
			c = 0
			saveFrame = 0

		key = cv2.waitKey(5)
		if key != -1:
	# ESC to quit
			if key == 27:
				break 
	# number of the object to toggle pause
			if key >= ord('0') and key <= ord('9'):
				saveFrame = 1 - saveFrame
				t = key - ord('0')

