
== Object Detection with SVM and OpenCV ==

00preprocessing.py	- here you can change "preprocess1", in order to have an optimal blurred image
01saveFrames.py		- here you save train images in "trainData" directory
02preprocessing1.py	- optional: you can save blurred images
03crossSVM.py		- cross validation: train the SVM using train samples and compare to the cross validation set. Here you can change C, gamma and preprocess*
04saveSVM.py		- once you are happy with your SVM, you can save it
05testSVM.py		- apply the trained SVM to the test set
06webcamSVM.py		- apply the trained SVM directly to the webcam for production

crossData		- this directory contains all the cross validation set
README.txt		- this file
testData		- this directory contains all the test set
trainData		- this directory contains all the train set
webCAM-Framesa		- in this directory frames of the webcam are saved

It is advised to split your samples in three sets:
- 60% for the Training Set
- 20% for the Cross Validation Set
- 20% for the Test Set

Coins are used as samples and the provided SVM is not good.

