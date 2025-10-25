<h1 align = "center">Dataset Generation Guide with Snap!</h1>

A **dataset** is a structured collection of labeled images used to train and evaluate deep learning models. Each image in the dataset is associated with a specific label or in the case of object detection, bounding boxes that define the location of objects within the image and their labels.

Creating a high-quality dataset is the **foundation for retraining or fine-tuning** models like **DetectNet** and **ImageNet.** By collecting and labeling your own images, you ensure the model learns to recognize the exact objects and scenarios relevant to your project

<p align="center">
  <img src="/docs/images/dataset-generation-header.png" alt="Dataset Generation" height="90%" width="90%" />
</p>

To create your own object detection or image classification models, the first step is to build a dataset containing the objects you want to train on.

Two dedicated Snap! projects are provided to make this process simple and visual. You can use it to collect, label, and organize images directly for [**DetectNet**](/snap_blocks/projects/detectnet_dataset_generator.xml) or [**ImageNet**](/snap_blocks/projects/imagenet_dataset_generator.xml) training.


### Table of Contents

* [DetectNet Dataset Generation](#detectNet-dataset-generation)
* [ImageNet Dataset Generation](#imagenet-dataset-generation)

## DetectNet Dataset Generation

The [DetectNet Dataset Generation project](/snap_blocks/projects/detectnet_dataset_generator.xml) is a fully finished Snap! project, so you don’t need to modify or adjust any of its internal logic. Simply click the green flag (start button) to begin. For the best experience, it’s highly recommended to maximize the Snap! canvas to get a full view of all available controls.

### Initial Setup

This project is available in English and Spanish, allowing you to choose your preferred language at the start.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5bbc06a2-7059-4235-aa58-a4adcabe2dd6" alt="Dataset Generation" height="90%" width="90%" />
</p>

#### Connection Setup

After selecting the language, you’ll be prompted to enter the IP address of your Jetson WebSocket server. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/aef6af1d-2277-46da-ac20-d1f2d090b71a" alt="Dataset Generation" height="90%" width="90%" />
</p>

***Make sure your computer and the Jetson device are connected to the same local network.***

Next, input the communication port used by your Jetson server to establish the connection. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/a52b4631-ba25-46c4-9ac5-84daedd8ab01" alt="Dataset Generation" height="90%" width="90%" />
</p>

***If everything is okay, you’ll see a successful connection message. If not, you’ll have two more opportunities to try again.***

#### Dataset name Setup

After a successful connection, the next step will prompt you to enter the dataset name you want to create.

<p align="center">
  <img src="https://github.com/user-attachments/assets/02d69cc7-bdda-424c-87df-7c55f8e2b374" alt="Dataset Generation" height="90%" width="90%" />
</p>

If the dataset name already exists, any new images you capture will be added to that existing dataset. By the label name you can choose to include **new** or an **existing labels** with previously captured images. This flexibility allows you to **extend**, **continue**, or **start** from scratch with your dataset as needed.

***Be careful not to accidentally reuse an existing dataset name, as this could cause new images to be added to the wrong dataset.***

### Menu options

After finishing the setup, press the **Space Bar** to open the **menu options**. From this menu, you can select different actions depending on what you need to do.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ef5aec2f-216b-4d8b-b9e0-352d53d835ed" alt="Dataset Generation" height="90%" width="90%" />
</p>

The available options include:

* **Add Class Name:** Create a new class label to categorize your images.
* **Delete Class Name:** Remove a previously added class label.
* **Send Image:** Capture and send an image with a selected class name to the Jetson server.
* **Connect to WebSocket:** Update the WebSocket IP address and port in case they have changed.
* **Update Dataset Name:** Create a new dataset or switch to another existing one.

For this example, we’ll explore how to **add a class name** and **send an image** to the server.

The **Connect to WebSocket** and **Update Dataset Name** options work the same way as during the initial setup.

Keep in mind that the **Delete Class Name** option only removes labels you have added during the current session. These label names are stored **temporarily**, so if you restart the application the added label names will be lost. Make sure to note them down if you plan to work with multiple labels.

#### Add Class Name

Click on **Add Class**, and you’ll be prompted to enter the name of the class you’d like to register.

Type the desired class name and then press **Enter** or click the **check button** beside the text box to confirm.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8349bfb3-2a25-491e-babf-d30dbb42c7f7" alt="Dataset Generation" height="90%" width="90%" />
</p>

Once submitted, the application will notify you that the registration was successful.

If you want to add more labels, simply press the **Space Bar** again to open the menu and repeat the same process.

#### Send and Image to the Server

After adding the class label names for the objects you want to capture, navigate to the **Send Image** option.

A selection menu will appear where you’ll need to choose the **class name** of the object you plan to submit to the server.

<p align="center">
  <img src="https://github.com/user-attachments/assets/00e6467f-178d-42e6-8eec-c29314260c72" alt="Dataset Generation" height="90%" width="90%" />
</p>

Once you’ve selected the class name, your device’s **camera** will activate. You’ll see a **red button** in the center of the screen and when you’re ready, press this button to capture an image.


## ImageNet Dataset Generation
