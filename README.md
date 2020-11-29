# SSD Object Detection
This repository concerns multiple object detection in one shot using a SSD model based on a SSD MobileNet V2 architecture. <br>
What is SSD Object Detection? <br>
SSD is Single Shot Multi-Box Detector and the SSD MobileNet V2 is an architecture which is trained over the COCO dataset using TensorFlow API. This model is able to detect 90 different types of objects using only one single shot i.e the COCO dataset consists of 90 different classes. SSD object detection also is proven to be fatser in performance to R-CNN since the R-CNN is unable to detect multiple objects in a single shot. MobileNet is the base architecture for SSD, this architecture is sinle consisting of a single convolutional neural network architecture having several convolutional neural network layers used to detect for bounding box locations and ckassify them. This can be trained end to end. <br>
The TensorFlow object detection API provides an entire deep learning netowrk consisting of pretrained models referred to as the Model Zoo. These models help in solving varoius object detection problems. These of these models are trained on seperate datasets each based on a seperate architecture. In our case we have used a SSD model having a MobileNet architecture trained on the COCO dataset. <br>
To refer to the TensorFlow object detection API or download their model for illustration route here: (https://github.com/tensorflow/models/tree/master/research/object_detection) <br>
or <br>
git clone (https://github.com/tensorflow/models.git) <br>
Some important commands: <br>
o pip install TensorFlow==1.15 lxml pillow matplotlib jupyter contextlib2 cython tf_slim <br>
o 'protoc object_detection/protos/*.proto --python_out=' <br>
o 'python setup.py build' <br>
o 'python setup.py install' <br>
o 'pip install pycocotools' or 'pip install git+(https://github.com/philferriere/cocoa...^&subdirectory=PythonAPI') <br>

It is to be noted that the above mentioned commands and model reference are just the important necessary commands for any object detection model to be executed. <br>

In this repository, 
The ObjectDetection.ipynb file consists of the code for detection for multiple objects on a single capture from frame.  <br>
The ModelTrainingCollab file conists of the lengthy process of training the SSD MobileNet V2 architecture for it's implementation in object detection.
