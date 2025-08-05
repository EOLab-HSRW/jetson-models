<h1 align = "center">Jetson Model Manager X Snap! </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9056f44-5639-41bb-b2b1-2473cf0680e9" alt="Jetson X Snap!" />
</p>

## Deploying Deep Learning for Everyone

This repository provides a Python-based model manager designed for running AI inference models on a Jetson device. It leverages NVIDIA's [Jetson Inference Library](https://github.com/dusty-nv/jetson-inference) to manage deep learning models efficiently. Additionally, the manager sets up a WebSocket server, enabling remote clients to interact with the models and send images for real-time inference.

The main objective is to facilitate model interaction through [Snap!](https://snap.berkeley.edu/), a block-based programming language for educational and demonstrative purposes, the manager also supports connectivity from any programming language that supports WebSocket communication.

🔗 **Snap! to the `js` Extension Link:** [https://eolab-hsrw.github.io/jetson-models/snap_blocks/jetson.js](https://eolab-hsrw.github.io/jetson-models/snap_blocks/jetson.js)

---

## Installation and Setup

### System Requirements
Ensure you have the following dependencies installed before proceeding:
- NVIDIA Jetson device (Nano, Xavier, or AGX series)
- Ubuntu-based OS (JetPack recommended)
- Python 3.6+

### 1. Clone the Repository with Submodules

This project includes **Git submodules**. To properly clone the repository, use the following command:

```sh
git clone https://github.com/EOLab-HSRW/jetson-models.git --recurse-submodules
```

This ensures that all submodule folders are initialized and downloaded automatically.

**Updating Submodules**

If any submodule gets updated in the repository, you can sync your local copies using:

```sh
git submodule update --init --recursive
```

Or, to pull the latest changes from submodules along with the main repository:

```sh
git pull --recurse-submodules
```

### 2. Update System Packages

Before installing dependencies, update and upgrade your system:

```sh
sudo apt-get update
sudo apt-get upgrade -y
```

### 3. Install PyTorch and TorchVision (for Jetson)

First, install gdown to download the correct versions of PyTorch and TorchVision:

```sh
pip install gdown

```

Then, download and install **PyTorch:**

```sh
gdown https://drive.google.com/uc?id=1TqC6_2cwqiYacjoLhLgrZoap6-sVL2sd
pip3 install torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl
```

Download and install **TorchVision:**

```sh
gdown https://drive.google.com/uc?id=1C7y6VSIBkmL2RQnVy8xF9cAnrrpJiJ-K
pip3 install torchvision-0.11.0a0+fa347eb-cp36-cp36m-linux_aarch64.whl
```

To ensure compatibility with Jetson devices, install the required system packages:

```sh
sudo apt-get install -y cmake build-essential python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
```

Then, install necessary Python packages:

```sh
sudo -H pip3 install future
```

```sh
sudo pip3 install -U --user wheel mock pillow
```

```sh
sudo -H pip3 install --upgrade setuptools
```

```sh
sudo -H pip3 install cython
```

### 4. Install Model Manager Dependencies

Navigate to the project directory and install all required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

### Configure the WebSocket Server

To configure and start the WebSocket server by running `main.py` with optional arguments:

```bash
python3 main.py --ip=127.0.0.1 --port=5000 --delete_datasets=False --debug=False
```

#### Parameters

| Argument                     | Description                                                         | Default   |
|:-----------------------------|:--------------------------------------------------------------------|:----------|
| **--ip** (str)               | IP address where the Jetson device is reachable on your network.    | 127.0.0.1 |
| **--port** (int)             | Port used for WebSocket communication.                              | 5000      |
| **--delete_datasets** (bool) | If True, deletes all created datasets before the server shuts down. | False     |
| **--debug** (bool)           | Enables detailed logging for tracing system interactions.           | False     |

*All arguments are optional; the defaults will be used if they’re not provided.*

## Inference with Snap!

To use the full capabilities of the Model Manager, you'll need to download the [customs Snap! blocks!](https://github.com/EOLab-HSRW/jetson-models/tree/main/snap_blocks) These blocks are currently available in English and Spanish. 

### Setup Instruncions

* **Download or clone the [Snap! GitHub Project](https://github.com/jmoenig/Snap)**

```bash
git clone https://github.com/jmoenig/Snap.git
```

*While Snap! can run from its official website, it cannot connect via WebSocket due to security restrictions. The online version uses HTTPS, which blocks non-secure WebSocket (ws://) connections.*

* **Run Snap! Locally**

After downloading, open the `snap.html` file from the project folder in your browser.

* **Import the Custom Blocks**

Load the downloaded Snap! blocks into your local Snap! session.

### Supported Models and their Usage

The system supports various pretrained deep learning models from Jetson Inference for tasks such as object detection, classification, pose estimation, and semantic segmentation. Each model can be launched and interacted with via Snap! using customized blocks.

| Model Name          | Python                                                                                 |
|:--------------------|:---------------------------------------------------------------------------------------|
| Object Detection    | [detectnet](https://github.com/EOLab-HSRW/jetson-models/blob/main/models/detectnet.py) |
| Image Recognition   | [imagenet](https://github.com/EOLab-HSRW/jetson-models/blob/main/models/imagenet.py)   |
| Segmentation        | [segnet](https://github.com/EOLab-HSRW/jetson-models/blob/main/models/segnet.py)       |
| Pose Estimation     | [posenet](https://github.com/EOLab-HSRW/jetson-models/blob/main/models/posenet.py)     |

### Connect to the server

By using the connect to Jetson block you have to provide the IP and the port of the WebSocket server. This block returns a WebSocket Object with allow the communication betweent the server and the client. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ad79b8e-7c2c-4225-9786-6d4e846d8973" alt="Connect to Jetson Block" />
</p>

Once connected using the `connect to Jetson` block, you can launch a detection model using the `send msg to socket with response` block. This block returns a model ID `(e.g., 1000)`, which is used to reference the launched model instance.

### Launch models

#### Launch a Object Detection Model - `detectnet`

<p align="center">
  <img src="https://github.com/user-attachments/assets/22580a60-f0ba-44c9-9c24-33ab439e2801" alt="Launch DetectNet Model" />
</p>

*Use the Snap! JSON extension blocks to construct the appropriate JSON request.*

#### JSON Payload Structure for detectnet /models/launch

The `send msg to socket with response` block expects a JSON object like this:

```json
{
  "command": "/models/launch",
  "model_name": "detectnet",
  "variant_name": "ssd-inception-v2",
  "threshold": 0.6,
  "overlay": "box,labels"
}
```

#### JSON Argument Reference

| Key           | Type    | Required  | Default          |
|:--------------|:--------|:----------|:-----------------|
| command       | string  | Yes       | —                |
| model_name    | string  | Yes       | —                |
| variant_name  | string  | No        | ssd-mobilenet-v2 |
| threshold     | float   | No        | 0.5              |
| overlay       | string  | No        | box,labels,conf  |

*You can omit optional fields if you're happy with the defaults. However, specifying them gives more control.*

#### Available Object Detection Variants

| Variant                    | variant_name Argument    |
|:---------------------------|:-------------------------|
| SSD-Mobilenet-v1           | ssd-mobilenet-v1         |
| SSD-Mobilenet-v2 (Default) | ssd-mobilenet-v2	        | 
| SSD-Inception-v2           | ssd-inception-v2         |
| TAO PeopleNet              | peoplenet                |
| TAO PeopleNet (pruned)     | peoplenet-pruned         |
| TAO DashCamNet             | dashcamnet	              | 
| TAO TrafficCamNet          | trafficcamnet            |
| TAO FaceDetect             | facedetect               |
| DetectNet-COCO-Dog         | coco-dog	                | 
| DetectNet-COCO-Bottle      | coco-bottle              |
| DetectNet-COCO-Chair       | coco-chair               |
| DetectNet-COCO-Airplane    | coco-airplane            |
| ped-100                    | pednet	                  | 
| multiped-500               | multiped                 |
| facenet-120                | facenet                  |

#### Launch a Image Recognition Model - `imagenet`

<p align="center">
  <img src="https://github.com/user-attachments/assets/1d198b37-04aa-4bcd-a55d-f1b00ea68fc4" alt="Launch ImageNet Model" />
</p>

*Use the Snap! JSON extension blocks to construct the appropriate JSON request.*

#### JSON Payload Structure for imagenet /models/launch

The `send msg to socket with response` block expects a JSON object like this:

```json
{
  "command": "/models/launch",
  "model_name": "imagenet",
  "variant_name": "alexnet",
  "topK": 2
}
```

#### JSON Argument Reference

| Key          | Type    | Required  | Default   |
|:-------------|:--------|:----------|:----------|
| command      | string  | Yes       | —         |
| model_name   | string  | Yes       | —         |
| variant_name | string  | No        | googlenet |
| topK         | Integer | No        | 1         |

*You can omit optional fields if you're happy with the defaults. However, specifying them gives more control.*

#### Available Image Recognition Variants

| Variant             | variant_name Argument   |
|:--------------------|:------------------------|
| AlexNet             | alexnet                 |
| GoogleNet (Default) | googlenet	              | 
| GoogleNet-12        | googlenet-12            |
| ResNet-18           | resnet-18               |
| ResNet-50           | resnet-50               |
| ResNet-101          | resnet-101	            | 
| ResNet-152          | resnet-152              |
| VGG-16              | vgg-16	                |
| VGG-19              | vgg-19		              | 
| Inception-v4        | inception-v4            |

#### Launch a Segmentation Model - `segnet`

<p align="center">
  <img src="https://github.com/user-attachments/assets/5d8402ff-b00e-4db8-81b6-7098964608f5" alt="Launch SegNet Model" />
</p>


*Use the Snap! JSON extension blocks to construct the appropriate JSON request.*

#### JSON Payload Structure for segnet /models/launch

The `send msg to socket with response` block expects a JSON object like this:

```json
{
  "command": "/models/launch",
  "model_name": "segnet",
  "variant_name": "fcn-resnet18-cityscapes-512x256",
  "filter_mode": "point",
  "alpha": 155.5,
  "ignore_class": "void",
  "visualize": "overlay,mask"
}
```

#### JSON Argument Reference

| Key          | Type    | Required  | Default                  |
|:-------------|:--------|:----------|:-------------------------|
| command      | string  | Yes       | —                        |
| model_name   | string  | Yes       | —                        |
| variant_name | string  | No        | fcn-resnet18-voc-320x320 |
| filter_mode  | string  | No        | linear                   |
| alpha        | float   | No        | 150.0                    |
| ignore_class | string  | No        | void                     |
| visualize    | string  | No        | overlay,mask             |

*You can omit optional fields if you're happy with the defaults. However, specifying them gives more control.*

#### Available Segmentation Variants

| Dataset                                                             | variant_name Argument             |
|:--------------------------------------------------------------------|:----------------------------------|
| [Cityscapes](https://www.cityscapes-dataset.com/)                   | fcn-resnet18-cityscapes-512x256   |
| [Cityscapes](https://www.cityscapes-dataset.com/)                   | fcn-resnet18-cityscapes-1024x512	| 
| [Cityscapes](https://www.cityscapes-dataset.com/)                   | fcn-resnet18-cityscapes-2048x1024 |
| [DeepScene](https://deepscene.cs.uni-freiburg.de/)                  | fcn-resnet18-deepscene-576x320    |
| [DeepScene](https://deepscene.cs.uni-freiburg.de/)                  | fcn-resnet18-deepscene-864x480    |
| [Multi-Human](https://lv-mhp.github.io/)                            | fcn-resnet18-mhp-512x320          | 
| [Multi-Human](https://lv-mhp.github.io/)                            | fcn-resnet18-mhp-512x320          |
| [Pascal VOC](https://github.com/paperswithcode/paperswithcode-data) | fcn-resnet18-voc-320x320	        |
| [Pascal VOC](https://github.com/paperswithcode/paperswithcode-data) | fcn-resnet18-voc-512x320		      | 
| [SUN RGB-D](https://rgbd.cs.princeton.edu/)                         | fcn-resnet18-sun-512x400          |
| [SUN RGB-D](https://rgbd.cs.princeton.edu/)                         | inception-v4                      |


#### Launch a Pose Estimation Model - `posenet`

<p align="center">
  <img src="https://github.com/user-attachments/assets/38ffff35-7612-45b2-925a-baff2f6cb697" alt="Launch PoseNet Model" />
</p>

*Use the Snap! JSON extension blocks to construct the appropriate JSON request.*

#### JSON Payload Structure for posenet /models/launch

The `send msg to socket with response` block expects a JSON object like this:

```json
{
  "command": "/models/launch",
  "model_name": "posenet",
  "variant_name": "resnet18-body",
  "overlay": "none",
  "threshold": 0.15
}
```

#### JSON Argument Reference

| Key          | Type    | Required  | Default       |
|:-------------|:--------|:----------|:--------------|
| command      | string  | Yes       | —             |
| model_name   | string  | Yes       | —             |
| variant_name | string  | No        | resnet18-body |
| overlay      | string  | No        | none          |
| threshold    | float   | No        | 0.15          |

*You can omit optional fields if you're happy with the defaults. However, specifying them gives more control.*

#### Available Pose Estimation Variants

| Variant                       | variant_name Argument   |
|:------------------------------|:------------------------|
| Pose-ResNet18-Body (Default)  | resnet18-body           |
| Pose-ResNet18-Hand            | resnet18-hand	          | 
| Pose-DenseNet121-Body         | densenet121-body        |
