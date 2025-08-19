<h1 align = "center">SegNet Usage Guide with Snap!</h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9056f44-5639-41bb-b2b1-2473cf0680e9" alt="Jetson X Snap!" />
</p>

## Connect to the server

By using the `connect to Jetson` block you have to provide the IP and the port of the WebSocket server. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ad79b8e-7c2c-4225-9786-6d4e846d8973" alt="Connect to Jetson Block" />
</p>

**Return values**
* This block returns a **`WebSocket Object`** that allow the communication betweent the server and the client.

## Launch a Segmentation Model - `segnet`

Once connected you can launch a segmentation model using the `send msg to socket with response` block.

<p align="center">
  <img src="https://github.com/user-attachments/assets/33f21e3b-b55e-4f40-a1f2-3796860bbb85" alt="Launch SegNet Model" />
</p>

**Return values**
* **Positive integer (e.g., `1000`)** → Successful model launch (model ID).
* `-1` → Internal error, such as trying to launch a model that is not downloaded.
* `0` → Invalid parameters (e.g., mistyped model name or wrong values).

*Use the Snap! JSON extension blocks to construct the appropriate JSON request or copy JSON Payload structure below.*

### JSON Payload Structure for segnet /models/launch

The `send msg to socket with response` block expects a JSON object like this:

```json
{
  "command": "/models/launch",
  "model_name": "segnet",
  "variant_name": "fcn-resnet18-voc-320x320",
  "filter_mode": "linear",
  "alpha": 150.0,
  "ignore_class": "void",
  "visualize": "overlay,mask"
}
```

### JSON Argument Reference

| Key          | Type    | Required  | Default                  |
|:-------------|:--------|:----------|:-------------------------|
| command      | string  | Yes       | —                        |
| model_name   | string  | Yes       | —                        |
| variant_name | string  | No        | fcn-resnet18-voc-320x320 |
| filter_mode  | string  | No        | linear                   |
| alpha        | float   | No        | 150.0                    |
| ignore_class | string  | No        | void                     |
| visualize    | string  | No        | overlay,mask             |

*You can omit the fields that are not required if you're happy with the defaults. However, specifying them gives more control.*

### Available Segmentation Variants

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

## Run the Launched Segmentation Model

To perform semantic segmentation using video input, you'll need to create a loop using the repeat until block that runs continuously until a specific condition is met.

Within this loop:

1. Capture a frame from the video using the `video on` block. 
2. Encode the captured image to Base64 format using the `encode base64 of` block.

   **Return Values**
   * **Base64 string** → Successful encoding.
   * `-1` → Error: input was not a valid image.
     
4. Send the encoded image to the Jetson server using the `send base64_img to socket to model with response` block. This block sends the image through the active WebSocket connection executing using a specific segmentation model.

   **Return Values**
   * **JSON with the segmented image and its segmentation information (e.g, ClassID, ClassLabel, PixelCount)** → Successful inference.
   * **Empty list `[]`** → One or more parameters were invalid (e.g., wrong model ID, invalid image, missing connection).
   
6. Display the result using the `draw segmentations` block. This block show the segmentation image into the canvas using the segmentation list.

   **Behavior**
   * If the segementation list is **non-empty**, the segmented image is showed into the canvas.
   * If the detection list is **empty**, the block does nothing (no error, just skipped).

<p align="center">
  <img src="https://github.com/user-attachments/assets/d1b49a7d-a0c6-4920-8c2e-968b0da22636" alt="Run SegNet Model" />
</p>

### JSON Response Structure

Here’s an example of the response returned from a successful inference using segnet:

```json
{
    "segmentation_info": [
        {
            "ClassID": 15,
            "ClassLabel": "person",
            "PixelCount": 13824
        }
    ],
    "image_data": "<encoded_base64_image>"
}
```

## Stop the Launched Segmentation Model

It’s important to stop models after execution to free up system resources and ensure optimal performance. You can do this using the `stop model to socket` block, which supports three modes of operation depending on your needs:

1. **Stop a Single Model by ID**

    Provide the specific `model_id` to stop a single model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/63e4e4ed-b867-4eaf-8dd3-4b61a99ae924" alt="Stop SegNet Model" />
</p>

 * On success, returns the ID of the stopped model.

2. **Stop Multiple Models Using a List**

    Provide a list of `model_ids` (e.g., `[1000,1002,1025]`).

<p align="center">
  <img src="https://github.com/user-attachments/assets/e752b5ab-72fe-4aee-b243-6fc60ec6adb4" alt="Stop SegNet Model by a list" />
</p>

  * On success, returns a list of the IDs of the stopped models.

3. **Stop All Running Models**

    Pass the string "ALL" to stop all active models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/391aa996-69e4-41c7-9c8e-56f8efaedb0c" alt="Stop all running models" />
</p>
  
  * On success, returns a list of all IDs of the models that were stopped.

**Return Values**
* **ID or List of IDs** → Successful stop operation.
* `0` → No models were found that matched the given ID(s), invalid `model_id` type, or no models were running.
* `-1` → Internal error (e.g., missing required JSON keys, exception while stopping).

