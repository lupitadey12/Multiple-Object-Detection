# SSD Object Detection
This repository concerns multiple object detection in one shot using a SSD model based on a SSD MobileNet V2 architecture. <br>
What is SSD Object Detection? <br>
SSD is Single Shot Multi-Box Detector and the SSD MobileNet V2 is an architecture which is trained over the COCO dataset using TensorFlow API. The SSD pretrained model is able to detect 90 different types of objects using only one single shot i.e the COCO dataset on which it is being trained on consists of 90 different classes hence the model learns to detect 90 objects. SSD object detection is extremely efficient and is proven to be faster in performance as compared to R-CNN since R-CNN cannot detect multiple objects in one go. MobileNet is the base architecture for the SSD model, this is a single convolutional neural network architecture having several convolutional neural network layers which are  used to detect for bounding box locations and classify them. The MobileNet architecture can be trained end to end. <br>
The TensorFlow object detection API provides an entire deep learning network consisting of pretrained models referred to as the Model Zoo. These models help in solving various object detection problems. Each of these models are trained on seperate datasets each based on a seperate architecture. In my case I have used a SSD model having a MobileNet architecture trained on the COCO dataset. <br>
To refer to the TensorFlow object detection API or download their model for illustration route here: (https://github.com/tensorflow/models/tree/master/research/object_detection) <br>
or <br>
git clone (https://github.com/tensorflow/models.git) <br>
In order to execute or run an object detection model there are some prerequisites and installations that are required hence it is to be noted that the below mentioned commands and model reference are just some important necessary commands. <br>
Commands: <br>
o pip install TensorFlow==1.15 lxml pillow matplotlib jupyter contextlib2 cython tf_slim <br>
o protoc object_detection/protos/*.proto --python_out= <br>
o python setup.py build <br>
o python setup.py install <br>
o pip install pycocotools or pip install git+(https://github.com/philferriere/cocoa...^&subdirectory=PythonAPI') <br>

This repository concerns multiple object detection in one shot using a SSD model based on a SSD MobileNet V2 architecture. The ObjectDetection.ipynb file consists of the code for detection for multiple objects on a single capture from frame. The ModelTrainingCollab file conists of the lengthy process of training the SSD MobileNet V2 architecture for it's implementation in object detection.
