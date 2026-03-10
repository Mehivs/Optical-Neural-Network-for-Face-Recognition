# SmartFace
## Introduction
SmartFace (SF) is a python implementation of a diffractive optical Siamese neural network, which can be used for face verification. https://ieeexplore.ieee.org/document/1467314.  Currently, it supports training an all-optical verifier on 2D facial dataset (e.g. AR database https://www2.ece.ohio-state.edu/~aleix/ARdatabase.html). 


## Install
no installation is required. Just copy the whole directory to where you want to use the package. Although you don't need to install SmartFace, you do need to install several common machine learning software which the SmartFace are rely on.

## dependence
* python3
* Pytorch >= 1.9
* numpy
* matplotlib
* Pillow
* cv2
* tqdm
* pandas
* sklearn
* scipy
* tensorboard


## Getting start
The SmartFace (SF), currently only supports a coherent light model.

The general idea is very simple. the input of the network is the face image, the output is a vector, which has a size equal to number of detectors you defined. The function of the network is to encode the face image into a lower dimension vector representation, so that you can calculated the distance between two face images with less computational resource. Ideally, given a pair of images (two image will pass through the same network twice), if they are from the same person (genuine pair), the L2 distance of the output vectors should be small. If they are from different persons (impostor), the distance should be large. In order to train the network behave in this way we need contrastive loss to encourage large distance for impostor and encourage small distance for genuine. Some math details will be provided in tutorial.ppt.

For each image in your dataset, the value of each pixel is treated as a light source (you can increase the resolution of your image by cv2.resize). Then the light propagates in free space, passes through the optical layer (first the substrate then the metalayer), free space again, and finally reaches the detector. As shown in the figure.
![model](images/model.jpg)

**caveat**:\
1.there are many pathes involved here. If it report something not found, make sure you make the path point to the right location. eg: the module_path in forward.py and main.py, the pathes in the config file. and the path you send to --config argrument. \
2. I try my best to make the code organized, but this is not a well developed user friendly software, make sure you know how to read the code, and understanding the structure of the code. So that you can know what you are simulation,and you are able to handle small bugs.\
3. the structure of the code:\
the python program you can run is in code/,
the modules that are imported by the python programs are in SmartFace/.
to run the python program, you need to specify the setup of your simulation in a .json file. (eg. how many layers, the physical size of the layers, the resolution of the layer)
To understand the json file, you need to goto SmartFace/data_util.py the dataclass Config. I made comment for each flag. Make sure your read each line of this dataclass. You will learn that if you set some flag to null will turnoff certain function. (eg. if the "training_dir": null, the training is off.) and the flag "num_layer" may not be what you thought, when go from free space into substrate, you are passing 1 layer. To simulate a matasurface with substrate you need num_layer = 2. if you have 2 matasurface with substrates, num_layer = 4.

**To run the forward simulation**:
you should be at the SmartFaceClass/ dir,
```
python code/forward.py --config "examples/configs/forward.json"
```

**To run the training simulation**:
you should be at the SmartFaceClass/ dir,
```
python code/main.py --config "examples/configs/train.json"
```
the output is stored in examples/output the dir name is same with your config name. So use different config name for different training.
In the output dir, we have a summary/ dir, where the tensorboard data is stored. you can open this file with tensorboard to track your training.
run ```tensorboard --logdir train``` to open tensorboard. as shown in the figure.
![tb](images/tensorboard.jpg)

**To run the test simulation**:
you should be at the SmartFaceClass/ dir,
```
python code/main.py --config "examples/configs/test.json"
```
the output is stored in examples/output the dir name is same with your config name. So use different config name for different test.
