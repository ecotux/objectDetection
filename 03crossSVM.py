
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

	dirTrain = 'trainData/'
	dirCross = 'crossData/'

	params = dict( kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_C_SVC, C = C, gamma = gamma )

# Loading Training Set
	print "Loading Training Set..."
	numTrainSamples = len([name for name in os.listdir(dirTrain)])
	trainSamples = np.empty( (numTrainSamples, yRes, xRes, 3), dtype = np.uint8 )
	targets = np.empty( numTrainSamples, dtype = np.float32 )

	for i, nameFile in enumerate(os.listdir(dirTrain)):
		match1=re.search(r"object(\d+)",nameFile)
		if match1:
			trainSamples[i] = cv2.imread(dirTrain+nameFile)
			targets[i] = np.float32(match1.group(1))

# Preprocessing Training Set
	print 'Preprocessing Training Set...'
	trainSet = np.array([preprocess2(preprocess1(trainSamples[i])) for i in np.ndindex(trainSamples.shape[:1])])

# Training
	print 'Training SVM...'
	model = cv2.SVM()
	model.train(trainSet, targets, params = params)

# Loading Cross Validation Set
	print "Loading Cross Validation Set..."
	numCrossSamples = len([name for name in os.listdir(dirCross)])
	crossSamples = np.empty( (numCrossSamples, yRes, xRes, 3), dtype = np.uint8 )
	crossTarget = np.empty( numCrossSamples, dtype = np.float32 )

	for i, nameFile in enumerate(os.listdir(dirCross)):
		match1=re.search(r"object(\d+)",nameFile)
		if match1:
			crossSamples[i] = cv2.imread(dirCross+nameFile)
			crossTarget[i] = np.float32(match1.group(1))

# Preprocessing Cross Validation Set
	print 'Preprocessing Cross Validation Set...'
	crossSet = np.array([preprocess2(preprocess1(crossSamples[i])) for i in np.ndindex(crossSamples.shape[:1])])

# Prediction
	print 'Prediction...'
	pred = model.predict_all(crossSet)
	resp = pred.reshape(1,-1)[0]

# Output
	print crossTarget
	print resp

for i in range(nObjects):
	print "Object", i, ": success",
	print float(np.sum(np.logical_and((resp == crossTarget),(crossTarget==i))))/np.sum(crossTarget==i)

