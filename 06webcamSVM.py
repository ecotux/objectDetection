
import cv2
import numpy as np

import os
import re

#############################################
#
# Gray magic:
# - the value of "C"
# - the value of "gamma"
# - the functions "preprocess*"
# 

C = 20
gamma = 0.0005

#
# blurring image
#

def preprocess1(data):
	img = cv2.GaussianBlur(data, (5,5), 0) 
	img = cv2.bilateralFilter(img,9,75,75)
	img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
	return img

#
# data feature extraction
#

def preprocess2(data):
	YCrCb = cv2.cvtColor(data, cv2.COLOR_BGR2YCR_CB)
	normalized = np.ravel(np.float32(YCrCb)) / 255.0
	return normalized[76800:]



#############################################
# 
#	Main
#
#############################################

if __name__ == '__main__':

	dirWebCam = 'webCAM-Frames//'

	cv2.namedWindow( "WebCAM 0", 1 )
	capture = cv2.VideoCapture(0)
	capture.set(3,320)
	capture.set(4,240)

# Loading SVM
	print 'Loading SVM...'
	model = cv2.SVM()
	model.load("objectSVM.xml")

# Processing WebCam
	print 'Processing WebCam...'
	showResp = 0
	i = -1
	while True:
		i += 1
		ret, src = capture.read()

		key = cv2.waitKey(5)
		if key != -1:
			if key == ord('p'):
				print 'Preprocessing frame...',
				test = preprocess2(preprocess1(src))
				print 'Prediction object...',
				resp = int(model.predict(test))
				print resp
				showResp = 1

			if key == ord('q'):
				showResp = 0

			if key == 27: 
				break 

		if showResp == 1:
			cv2.putText(src, str(resp), (250, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255))
		cv2.imshow("WebCAM 0", src)
		cv2.imwrite("webCAM-Frames/frame"+"-"+str(i).zfill(5)+".png",src)

