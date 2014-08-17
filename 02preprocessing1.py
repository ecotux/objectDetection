
#
# Optional: save blurred images
#

import re
import os

import cv2

def preprocess1(data):
	img = cv2.GaussianBlur(data, (5,5), 0) 
	img = cv2.bilateralFilter(img,9,75,75)
	img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
	return img

inputDir = 'crossData/'
outputDir = 'PREcrossData/'

for nameFile in os.listdir(inputDir):
	match1=re.search(r"object(\d+)",nameFile)
	if match1:
		print nameFile
		src = cv2.imread(inputDir+nameFile)
		data = preprocess1(src)
		cv2.imwrite(outputDir+nameFile, data)

