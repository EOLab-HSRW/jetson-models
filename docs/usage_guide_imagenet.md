<h1 align = "center">ImageNet Usage Guide with Snap!</h1>

Image classification is one of the fundamental tasks in computer vision. Instead of detecting multiple objects and their locations, classification networks predict the most likely class of the entire image. For example, given an input picture, the network outputs probabilities for categories such as dog, cat, car, airplane, and so on.

<p align="center">
  <img src="/docs/images/imagenet_demostration.gif" alt="DetectNet demonstration" height="70%" width="70%" />
</p>

The imageNet model from the [Jetson Inference Library](https://github.com/dusty-nv/jetson-inference)  takes an image as input and produces:

* **ClassID:** Numeric identifier of the predicted category
* **ClassLabel:** Human-readable label (e.g., lab coat, dog, car)
* **Confidence:** Probability score of the prediction

[Imagenet](../models/imagenet.py) is available to use from Python and supports a variety of [pre-trained image classification networks](#available-image-recognition-variants-and-average-performance-in-jetson-nano) optimized with TensorRT for real-time performance on Jetson devices. The default model is GoogLeNet. ImageNet is trained on the ILSVRC ImageNet dataset, which includes [1000 object classes.](https://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt)

## Connect to the server

By using the `connect to Jetson` block you have to provide the IP and the port of the WebSocket server. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ad79b8e-7c2c-4225-9786-6d4e846d8973" alt="Connect to Jetson Block" />
</p>

**Return values**
* This block returns a **`WebSocket Object`** that allow the communication betweent the server and the client.

## Launch a Image Recognition Model - `imagenet`

Once connected you can launch a classification model using the `send msg to socket with response` block.

<p align="center">
  <img src="https://github.com/user-attachments/assets/91de3a99-4f5c-4845-a4de-703e49a7c056" alt="Launch ImageNet Model" />
</p>

**Return values**
* **Positive integer (e.g., `1000`)** → Successful model launch (model ID).
* `-1` → Internal error, such as trying to launch a model that is not downloaded.
* `0` → Invalid parameters (e.g., mistyped model name or wrong values).

*Use the Snap! JSON extension blocks to construct the appropriate JSON request or copy JSON Payload structure below*

### JSON Payload Structure for imagenet /models/launch

The `send msg to socket with response` block expects a JSON object like this:

```json
{
  "command": "/models/launch",
  "model_name": "imagenet",
  "variant_name": "googlenet",
  "topK": 1
}
```

### JSON Argument Reference

| Key          | Type    | Required  | Default   |
|:-------------|:--------|:----------|:----------|
| command      | string  | Yes       | —         |
| model_name   | string  | Yes       | —         |
| variant_name | string  | No        | googlenet |
| topK         | integer | No        | 1         |

*You can omit optional fields if you're happy with the defaults. However, specifying them gives more control.*

### Available Image Recognition Variants and Average Performance in Jetson Nano

| Variant             | variant_name Argument   | Jetson Nano |
|:--------------------|:------------------------|:------------|
| AlexNet             | alexnet                 | 28 FPS      |
| GoogleNet (Default) | googlenet	              | 28 FPS      | 
| GoogleNet-12        | googlenet-12            | 28 FPS      |
| ResNet-18           | resnet-18               | 28 FPS      |
| ResNet-50           | resnet-50               | 23 FPS      |
| ResNet-101          | resnet-101	            | 15 FPS      |
| ResNet-152          | resnet-152              | 11 FPS      |
| VGG-16              | vgg-16	                | 10 FPS      |
| VGG-19              | vgg-19		              | 9 FPS       |
| Inception-v4        | inception-v4            | 9 FPS       |


## Run the Launched Object Detection Model

To perform classification using video input, you'll need to create a loop using the repeat until block that runs continuously until a specific condition is met.

Within this loop:

1. Capture a frame from the video using the `video on` block. 
2. Encode the captured image to Base64 format using the `encode base64 of` block.

   **Return Values**
   * **Base64 string** → Successful encoding.
   * `-1` → Error: input was not a valid image.
     
4. Send the encoded image to the Jetson server using the `send base64_img to socket to model with response` block. This block sends the image through the active WebSocket connection executing using a specific detection model.

   **Return Values**
   * **List of predictions (JSON objects)** → Successful inference.
   * **Empty list `[]`** → One or more parameters were invalid (e.g., wrong model ID, invalid image, missing connection).
   
6. Display the result using the `draw predictions` block. This block show the labels on top of the image using the predictions list.

   **Behavior**
   * If the prediction list is **non-empty**, show the prediction's class labels.
   * If the prediction list is **empty**, the block does nothing (no error, just skipped).

<p align="center">
  <img src="https://github.com/user-attachments/assets/26efcdac-c0f9-4e62-a879-cacfadf5b25a" alt="Run ImageNet Model" />
</p>

### JSON Response Structure

Here’s an example of the response returned from a successful inference using imagenet:

```json
[
    {
        "predictions": {
            "ClassID": 617,
            "ClassLabel": "lab coat, laboratory coat",
            "Confidence": 0.36767578125
        }
    }
]
```

## Stop the Launched Classification Model

It’s important to stop models after execution to free up system resources and ensure optimal performance. You can do this using the `stop model to socket` block, which supports three modes of operation depending on your needs:

1. **Stop a Single Model by ID**

    Provide the specific `model_id` to stop a single model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f28c0343-1200-4b7f-967f-607040acd1a0" alt="Stop ImageNet Model" />
</p>

 * On success, returns the ID of the stopped model.

2. **Stop Multiple Models Using a List**

    Provide a list of `model_ids` (e.g., `[1000,1002,1025]`).

<p align="center">
  <img src="https://github.com/user-attachments/assets/e752b5ab-72fe-4aee-b243-6fc60ec6adb4" alt="Stop ImageNet Model by a list" />
</p>

  * On success, returns a list of the IDs of the stopped models.

3. **Stop All Running Models**

    Pass the string "ALL" to stop all active models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/391aa996-69e4-41c7-9c8e-56f8efaedb0c" alt="Stop ImageNet Model by a list" />
</p>
  
  * On success, returns a list of all IDs of the models that were stopped.

**Return Values**
* **ID or List of IDs** → Successful stop operation.
* `0` → No models were found that matched the given ID(s), invalid `model_id` type, or no models were running.
* `-1` → Internal error (e.g., missing required JSON keys, exception while stopping).




