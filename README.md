<p align="center">
  <img src="/docs/images/deep-vision-with-jetson-and-snap!-header.png" alt="Deep vision with Jetson and Snap!" />
</p>

## Deploying Deep Learning — For Everyone to Use

This repository provides a Python-based model manager designed for running AI inference models on a Jetson device. It leverages NVIDIA's [Jetson Inference Library](https://github.com/dusty-nv/jetson-inference) to manage deep learning models efficiently. Additionally, the manager sets up a WebSocket server, enabling remote clients to interact with the models and send images for real-time inference.

The main objective is to facilitate model interaction through [Snap!](https://snap.berkeley.edu/), a block-based programming language for educational and demonstrative purposes, the manager also supports connectivity from any programming language that supports WebSocket communication.

Supported DDN vision models include [`DetectNet`](docs/usage_guide_detectnet.md) for object detection, [`ImageNet`](docs/usage_guide_imagenet.md) for image classification, [`PoseNet`](docs/usage_guide_posenet.md) for pose estimation and [`SegNet`](docs/usage_guide_segnet.md) for semantic segmentation. Examples are provided for live streaming from a camera into Snap!.

🔗 **Snap! to the `js` Extension Link:** [https://eolab-hsrw.github.io/jetson-models/snap_blocks/jetson.js](https://eolab-hsrw.github.io/jetson-models/snap_blocks/jetson.js)

🔗 **Pruebas de conocimiento preTest 5to B:** [Enlace](https://docs.google.com/forms/d/e/1FAIpQLSec76B3_7rMegAj_ptxsbGqMXJY9vsZeZF19iO0GA8QigUCbg/viewform?usp=header)

🔗 **Pruebas de conocimiento posTest 5to B:** [Enlace](https://docs.google.com/forms/d/e/1FAIpQLSdp5bRcrpLyuO0i6UCrwxLf5TGHCtM8yHE73QqVWQsIlVoe4Q/viewform?usp=header)

🔗 **Encuesta de funcionalidad 5to B:** [Enlace](https://docs.google.com/forms/d/e/1FAIpQLSdw1aE7ojKnWcKHoZk_pn780lpWEQSRoh0mGmSfyVoiomyYwA/viewform?usp=header)

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

## Server Usage

### Configure the WebSocket Server

To configure and start the WebSocket server by running `main.py` with optional arguments:

```bash
python3 main.py --ip=127.0.0.1 --port=5000 --delete_datasets=False --debug=False
```

### Parameters

| Argument                     | Description                                                         | Default   |
|:-----------------------------|:--------------------------------------------------------------------|:----------|
| **--ip** (str)               | IP address where the Jetson device is reachable on your network.    | 127.0.0.1 |
| **--port** (int)             | Port used for WebSocket communication.                              | 5000      |
| **--delete_datasets** (bool) | If True, deletes all created datasets before the server shuts down. | False     |
| **--debug** (bool)           | Enables detailed logging for tracing system interactions.           | False     |

*All arguments are optional; the defaults will be used if they’re not provided.*

## Inference with Snap!

You can run inferences visually using Snap! and interact with all supported models through block-based programming.  

Follow the setup instructions below to get started.

### Setup Instructions

1. **Download or clone the [Snap! GitHub Project](https://github.com/jmoenig/Snap)**

```bash
git clone https://github.com/jmoenig/Snap.git
```

*While Snap! can run from its [official website](https://snap.berkeley.edu/), it cannot connect via WebSocket because the online version uses HTTPS, which blocks non-secure WebSocket (ws://) connections.*

2. **Run Snap! Locally**

After downloading the project, open the snap.html file from the project folder in your browser.

3. **Import the Custom Blocks**

Download the [custom Snap! blocks](snap_blocks/) and load them into your local Snap! session.
(Available in English and Spanish.)

### Supported Models and their Usage

The system supports various pretrained deep learning models from Jetson Inference for tasks such as object detection, classification, pose estimation, and semantic segmentation. Each model can be launched and interacted with via Snap! using customized blocks.

| Model Name          | Python Source                       | Usage Guide with Snap!                           | 
|:--------------------|:------------------------------------|:------------------------------------------------ | 
| Object Detection    | [detectnet.py](models/detectnet.py) | [DetectNet Guide](docs/usage_guide_detectnet.md) |
| Image Recognition   | [imagenet.py](models/imagenet.py)   | [ImageNet Guide](docs/usage_guide_imagenet.md)   |
| Segmentation        | [segnet.py](models/segnet.py)       | [SegNet Guide](docs/usage_guide_segnet.md)       |
| Pose Estimation     | [posenet.py](models/posenet.py)     | [PoseNet Guide](docs/usage_guide_posenet.md)     |
