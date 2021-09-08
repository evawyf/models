<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/logo_project.png">
</p>

# Welcome to TensorFlow Model Garden Project Example - YOLOv3 :sunglasses:

[![YOLOv3](http://img.shields.io/badge/Paper-arXiv.1804.02767-B3181B?logo=arXiv)](https://arxiv.org/abs/1804.02767)
[![TFMG Tutorial](https://img.shields.io/badge/TFMG%20Tutorial-YOLO%20v3-F9AB00?logo=googlecolab)](https://githubtocolab.com/tf-models/readthedocs/blob/main/TFMG_Project_Tutorial_(v6).ipynb)
[![TensorFlow 2.5](https://img.shields.io/badge/TensorFlow-2.5-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB?logo=python)](https://www.python.org/downloads/release/python-379/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5-5C3EE8?logo=opencv)](https://opencv.org/)
<!-- [![TFDS 4.4](https://img.shields.io/badge/TF%20Datasets-4.4-FF6F00?logo=tensorflow)](https://www.tensorflow.org/datasets/overview) -->
<!-- [![Keras 2.5](https://img.shields.io/badge/Keras-2.5-D00000?logo=keras)](https://keras.io/) -->
<!-- [![GitHub](https://img.shields.io/badge/Up%20to%20Date-passing-green?logo=github)]() -->

Implement YOLOv3 by following TensorFlow Model Garden components. Support image and video detection tasks running on CPU / GPU.

<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/video_times_square.gif">
</p>

## TFMG Tutorial Colab and Implement Details

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/tf-models/readthedocs/blob/main/TFMG_Project_Tutorial_(v6).ipynb)


### TFMG Components

| Folders      | Required | Description             |
|-------------|----------|-------------------------------------------|
| modeling | yes      | Darknet-53 architecture, YOLO prediction base blocks and models.     |
| ops      | yes      | Operations for bounding boxes and non-maximum suppression (NMS).    |
| losses      | yes      | Loss functions.    |
| dataloaders | yes      | Decoders and parsers for your data pipeline; functions for downloading pretrained weigths.     |
| tasks       | yes      | Tasks for running the model. Tasks are essentially the main driver for training and evaluating the model.     |
| configs     | yes      | The  config  files for the task class to train and evaluate the model.      |
| common      | yes      | Registry imports. The tasks and configs need to be registered before execution.     |
| utils       | no       | Utility functions for draw boxes and video frame interface. |
| demos       | no       | Files needed to create a Jupyter Notebook/Google Colab demo of the model. |

### YOLOv3 Model

![]()

### Useful Tutorials

- TensorFlow / Keras Guide - Custom Layers and Models: 
    - https://keras.io/api/
    - https://www.tensorflow.org/guide/keras/custom_layers_and_models
- TensorFlow Core API document: 
    - https://www.tensorflow.org/api_docs/python/tf
- A Concise Handbook of TensorFlow 2: 
    - https://tf.wiki/en/


## TODO list

https://github.com/tf-models/readthedocs/issues/1

## Installation

[![TensorFlow 2.5](https://img.shields.io/badge/TensorFlow-2.5-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0)
[![TFDS 4.4](https://img.shields.io/badge/TF%20Datasets-4.4-FF6F00?logo=tensorflow)](https://www.tensorflow.org/datasets/overview)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB?logo=python)](https://www.python.org/downloads/release/python-379/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5-5C3EE8?logo=opencv)](https://opencv.org/)

### Requirements

- Python 3.7+
- OpenCV 

```
pip install -r requirements.txt
```

### TensorFlow2 Model Garden 

- TensorFlow 2.0+
- Model Garden
- TensorFlow Datasets


```
pip install -q tf-models-nightly tfds-nightly
```

## Download Pretrained Weights

Download pretrained weights from author's official source: 

- Darknet GitHub: https://github.com/pjreddie/darknet
- Joseph Chet Redmon's website: https://pjreddie.com

```
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O ./data/yolov3.weights
```

## Datasets

- COCO: 
    - https://www.tensorflow.org/datasets/catalog/coco
- VOC: 
    - https://www.tensorflow.org/datasets/catalog/voc


## Detection Task

Support image detection and video/webcamera detection. 

### Image File

```
# detect image
python detect.py --image ./data/dog.jpg 
python detect.py --image ./data/street.jpg  
```

<p align="center">
    <img src="outputs/output_dog.jpg">
</p>


### Camera 

```
# detect camera
python detect_video.py --video False
```


### Video File 

```
# detect video file
python detect_video.py --video_path ./data/times_square.mp4 --video True
```

<p align="center">
    <img src="https://raw.githubusercontent.com/tf-models/readthedocs/main/official/projects/yolo3/assets/video_times_square.gif">
</p>




## Training / Fine-tuning Task

```
python train.py --
```


## Command Line Args Reference

```
# detect.py

# video_detect.py

# train.py

```

## Tips

Avoid to push large files. First, run: 
```
find . -size +50M | cat >> .gitignore
```
Then, 
```
git add .
git commit -m "updates"
git push 
```
