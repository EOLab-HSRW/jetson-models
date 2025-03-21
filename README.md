<h1 align = "center">Jetson Model Manager X Snap! </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9056f44-5639-41bb-b2b1-2473cf0680e9" alt="Jetson X Snap!" />
</p>

## Overview

This repository provides a Python-based model manager designed for running AI inference models on a Jetson device. It leverages NVIDIA's [`Jetson Inference Library`](https://github.com/dusty-nv/jetson-inference) to manage deep learning models efficiently. Additionally, the manager sets up a WebSocket server, enabling remote clients to interact with the models and send images for real-time inference.

While the main objective is to facilitate model interaction through [Snap!](https://snap.berkeley.edu/), a block-based programming language for educational and demonstrative purposes, the manager also supports connectivity from any programming language that supports WebSocket communication. This makes it highly flexible for various applications.

### Features:
- Simplifies the management and execution of AI models on Jetson devices.
- Supports remote interaction via WebSocket.
- Enables real-time inference with various deep learning models.
- Optimized for educational and research environments.

🔗 **Snap! to the `js` Extension Link:** [`https://eolab-hsrw.github.io/jetson-models/snap_blocks/jetson.js`](https://eolab-hsrw.github.io/jetson-models/snap_blocks/jetson.js)

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

### 1. Configure the WebSocket Server

Before running the model manager, update the **host IP** in the **start_server** function located inside the `model_manager.py` file.

* Open the file:

```bash
cd model_manager/model_manager.py
```

* Locate the function `start_server` and modify the **host** variable with your Jetson device IP:

```python
async def start_server(self) -> None:
    """Start the WebSocket server"""

    print("WebSocket server is starting...")
    try:
        # Set the host IP
        host = "0.0.0.0"  # Change this to your desired Jetson IP
        port = 5000

        # Start the WebSocket server
        self.server = await websockets.serve(self.handle_client, host, port)
        print(f"Server running on ws://{host}:{port}")    
        await self.server.wait_closed()
        
    except Exception as e:
        print(f"Error starting the server: {e}")
    finally:
        print("Server shutdown.")
```

Make sure to replace "0.0.0.0" with the actual IP address where you want the server to be accessible.

### 2. Start the Model Manager

Once the host IP is set, execute the main.py file to start the model manager:

```sh
python3 main.py
```

### 3.  Connect to the WebSocket Server

* The server will be accessible at:

```python3
uri = "ws://<YOUR_HOST_IP>:5000"
websocket = await websockets.connect(uri)
```

* You can connect to the server using any WebSocket-compatible client.
* If using Snap!, ensure to use the **link to** block.
