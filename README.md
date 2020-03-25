# SSD_PyTorch

### Overview
- [SSD: Single Shot MultiBox Object Detector](http://arxiv.org/abs/1512.02325), this is a simple 
experiment to study the impact of different backbones on Performance.

This project is based on [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).Thanks.


### Table of contents
1. [About SSD](#about-ssd)
2. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pre-trained weights](#download-pre-trained-weights)
    * [Download COCO2014](#download-coco2014)
    * [Download VOC2007](#download-voc2007voc2012)
3. [Usage](#usage)
    * [Train](#train)
    * [Backbone](#backbone)
    * [Evaluation](#evaluation)
4. [Demos](#demos)
    * [Download a pre-trained network](#download-a-pre-trained-network)
    * [Try the demo notebook](#try-the-demo-notebook)
    * [Try the webcam demo](#try-the-webcam-demo)
5. [TODO](#todo)
6. [Credit](#credit) 

### About SSD
We present a method for detecting objects in images using a single deep neural network. 
Our approach, named SSD, discretizes the output space of bounding boxes into a set of 
default boxes over different aspect ratios and scales per feature map location. 
At prediction time, the network generates scores for the presence of each object 
category in each default box and produces adjustments to the box to better match 
the object shape. Additionally, the network combines predictions from multiple 
feature maps with different resolutions to naturally handle objects of various sizes. 
Our SSD model is simple relative to methods that require object proposals because it 
completely eliminates proposal generation and subsequent pixel or feature resampling 
stage and encapsulates all computation in a single network. This makes SSD easy to 
train and straightforward to integrate into systems that require a detection component. 
Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has 
comparable accuracy to methods that utilize an additional object proposal step and is much 
faster, while providing a unified framework for both training and inference. 
Compared to other single stage methods, SSD has much better accuracy, even with 
a smaller input image size. For 300×300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on 
a Nvidia Titan X and for 500×500 input, SSD achieves 75.1% mAP, outperforming a comparable 
state of the art Faster R-CNN model. Code is available at [this https URL](https://github.com/weiliu89/caffe/tree/ssd)) .

### Installation

#### Clone and install requirements
```bash
$ git clone https://github.com/Lornatang/SSD-PyTorch.git
$ cd SSD-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pre-trained weights
- First Download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `SSD_PyTorch/weights` dir:
```bash
$ mkdir weights
$ cd weights
$ wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```
#### Download COCO2014
```bash
# specify a directory for dataset to be downloaded into, else default is ~/data/
$ bash data/scripts/get_coco2014_dataset.sh
```

#### Download VOC2007+VOC2012
```bash
# specify a directory for dataset to be downloaded into, else default is ~/data/
$ bash data/scripts/get_voc_dataset.sh # <directory>
```

- Support [Visdom](https://github.com/facebookresearch/visdom) for real-time loss visualization during training!

To use Visdom in the browser:
```bash
# First install Python server and client
$ pip3 install visdom
# Start the server (probably in a screen or tmux)
$ python3 -m visdom.server
```
* Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).
- Note: For training, we currently support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://mscoco.org/), and aim to add [ImageNet](http://www.image-net.org/) support soon.

### Usage

#### Train
- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```bash
$ python3 train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

#### BackBone

------------------------

#### Evaluation
To evaluate a trained network:

```bash
$ python3 eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  

### Demos

Use a pre-trained SSD network for detection

#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models:
    * SSD300 trained on VOC0712 (newest PyTorch weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 trained on VOC0712 (original Caffe weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth
- Our goal is to reproduce this table from the [original paper](http://arxiv.org/abs/1512.02325)


#### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run):
    `jupyter notebook`

    2. If using [pip](https://pypi.python.org/pypi/pip):

```bash
# make sure pip is upgraded
$ pip3 install --upgrade pip
# install jupyter notebook
$ pip3 install jupyter
# Run this inside SSD_PyTorch
$ jupyter notebook
```

- Now navigate to `demo/demo.ipynb` at http://localhost:8888 (by default) and have at it!

#### Try the webcam demo
- Works on CPU (may have to tweak `cv2.waitkey` for optimal fps) or on an NVIDIA GPU
- This demo currently requires opencv2+ w/ python bindings and an onboard webcam
  * You can change the default webcam in `demo/live.py`
- Install the [imutils](https://github.com/jrosebr1/imutils) package to leverage multi-threading on CPU:
  * `pip3 install imutils`
- Running `python3 -m demo.live` opens the webcam and begins detecting!

### TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [x] Support for the MS COCO dataset
  * [x] Support for other backbones

### Credit

#### SSD: Single Shot MultiBox Detector
_Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg_<br>

**Abstract** <br>
We present a method for detecting objects in images using a single deep neural network. 
Our approach, named SSD, discretizes the output space of bounding boxes into a set of 
default boxes over different aspect ratios and scales per feature map location. 
At prediction time, the network generates scores for the presence of each object 
category in each default box and produces adjustments to the box to better match 
the object shape. Additionally, the network combines predictions from multiple 
feature maps with different resolutions to naturally handle objects of various sizes. 
Our SSD model is simple relative to methods that require object proposals because it 
completely eliminates proposal generation and subsequent pixel or feature resampling 
stage and encapsulates all computation in a single network. This makes SSD easy to 
train and straightforward to integrate into systems that require a detection component. 
Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has 
comparable accuracy to methods that utilize an additional object proposal step and is much 
faster, while providing a unified framework for both training and inference. 
Compared to other single stage methods, SSD has much better accuracy, even with 
a smaller input image size. For 300×300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on 
a Nvidia Titan X and for 500×500 input, SSD achieves 75.1% mAP, outperforming a comparable 
state of the art Faster R-CNN model. Code is available at [this https URL](https://github.com/weiliu89/caffe/tree/ssd)) .

[[Paper]](http://arxiv.org/abs/1512.02325)[[Authors' Implementation (Caffe)]](https://github.com/weiliu89/caffe/tree/ssd)

```
@inproceedings{liu2016ssd,
  title = {{SSD}: Single Shot MultiBox Detector},
  author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
  booktitle = {ECCV},
  year = {2016}
}
```