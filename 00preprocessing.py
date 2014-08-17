
import cv2
import numpy as np

def preprocess1(data):
	img = cv2.GaussianBlur(data, (5,5), 0) 
	img = cv2.bilateralFilter(img,9,75,75)
	img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
	return img


######## Main

if __name__ == '__main__':

	cv2.namedWindow("Orig", 1)
	cv2.namedWindow("Preprocessing", 1)

	capture = cv2.VideoCapture(0)
	capture.set(3,320)
	capture.set(4,240)

	while True:
		ret, src = capture.read()
		preImg = preprocess1( src )

		cv2.imshow("Orig", src)
		cv2.imshow("Preprocessing", preImg)

		key = cv2.waitKey(5)
		if key == 27:
			break

