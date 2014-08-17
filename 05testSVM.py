
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

	nObjects = 4

	xRes = 320
	yRes = 240

	dirTest = 'testData/'

# Loading SVM
	print 'Loading SVM...'
	model = cv2.SVM()
	model.load("objectSVM.xml")

# Loading Test Set
	print "Loading Test Set..."
	numTestSamples = len([name for name in os.listdir(dirTest)])
	testSamples = np.empty( (numTestSamples, yRes, xRes, 3), dtype = np.uint8 )
	testTarget = np.empty( numTestSamples, dtype = np.float32 )

	for i, nameFile in enumerate(os.listdir(dirTest)):
		match1=re.search(r"object(\d+)",nameFile)
		if match1:
			testSamples[i] = cv2.imread(dirTest+nameFile)
			testTarget[i] = np.float32(match1.group(1))

# Preprocessing Test Set
	print 'Preprocessing Test Set...'
	testSet = np.array([preprocess2(preprocess1(testSamples[i])) for i in np.ndindex(testSamples.shape[:1])])

# Prediction
	print 'Prediction...'
	pred = model.predict_all(testSet)
	resp = pred.reshape(1,-1)[0]

# Output
	print testTarget
	print resp

for i in range(nObjects):
	print "Object", i, ": success",
	print float(np.sum(np.logical_and((resp == testTarget),(testTarget==i))))/np.sum(testTarget==i)

