<h1 align = "center">DetectNet Usage Guide with Snap!</h1>

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

## Launch Object Detection Model - `detectnet`

Once connected you can launch a detection model using the `send msg to socket with response` block.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a445f14a-448b-41a5-a789-ee0b28d9b91a" alt="Launch DetectNet Model" />
</p>

**Return values**
* **Positive integer (e.g., `1000`)** → Successful model launch (model ID).
* `-1` → Internal error, such as trying to launch a model that is not downloaded.
* `0` → Invalid parameters (e.g., mistyped model name or wrong values).

*Use the Snap! JSON extension blocks to construct the appropriate JSON request or copy JSON Payload structure below.*

### JSON Payload Structure for detectnet /models/launch

The `send msg to socket with response` block expects a JSON object like this:

```json
{
  "command": "/models/launch",
  "model_name": "detectnet",
  "variant_name": "ssd-mobilenet-v2",
  "threshold": 0.5,
  "overlay": "box,labels,conf"
}
```

### JSON Argument Reference

| Key           | Type    | Required  | Default          |
|:--------------|:--------|:----------|:-----------------|
| command       | string  | Yes       | —                |
| model_name    | string  | Yes       | —                |
| variant_name  | string  | No        | ssd-mobilenet-v2 |
| threshold     | float   | No        | 0.5              |
| overlay       | string  | No        | box,labels,conf  |

*You can omit the fields that are not required if you're happy with the defaults. However, specifying them gives more control.*

### Available Object Detection Variants

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


## Run the Launched Object Detection Model

To perform object detection using video input, you'll need to create a loop using the repeat until block that runs continuously until a specific condition is met.

Within this loop:

1. Capture a frame from the video using the `video on` block. 
2. Encode the captured image to Base64 format using the `encode base64 of` block.

   **Return Values**
   * **Base64 string** → Successful encoding.
   * `-1` → Error: input was not a valid image.
     
4. Send the encoded image to the Jetson server using the `send base64_img to socket to model with response` block. This block sends the image through the active WebSocket connection executing using a specific detection model.

   **Return Values**
   * **List of detections (JSON objects)** → Successful inference.
   * **Empty list `[]`** → One or more parameters were invalid (e.g., wrong model ID, invalid image, missing connection).
   
6. Display the result using the `draw detection` block. This block renders bounding boxes on top of the image using the detection list.

   **Behavior**
   * If the detection list is **non-empty**, bounding boxes are drawn.
   * If the detection list is **empty**, the block does nothing (no error, just skipped)..

<p align="center">
  <img src="https://github.com/user-attachments/assets/8194aae6-9105-4975-a5ec-d1cc84f8b2ee" alt="Run DetectNet Model" />
</p>

### JSON Response Structure

Here’s an example of the response returned from a successful inference using detectnet:

```json
[
    {
        "detections": {
            "ClassID": 1,
            "ClassLabel": "person",
            "Confidence": 0.77587890625,
            "BoundingBox": {
                "Left": 0.0071,
                "Top": 0.2507,
                "Right": 0.8936,
                "Bottom": 0.9971
            }
        }
    }
]
```

## Stop the Launched Object Detection Model

It’s important to stop models after execution to free up system resources and ensure optimal performance. You can do this using the `stop model to socket` block, which supports three modes of operation depending on your needs:

1. **Stop a Single Model by ID**

    Provide the specific `model_id` to stop a single model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c2511cb9-09a5-429e-9098-4a03806a0a89" alt="Stop DetectNet Model" />
</p>

 * On success, returns the ID of the stopped model.

2. **Stop Multiple Models Using a List**

    Provide a list of `model_ids` (e.g., `[1000,1002,1025]`).

<p align="center">
  <img src="https://github.com/user-attachments/assets/e752b5ab-72fe-4aee-b243-6fc60ec6adb4" alt="Stop DetectNet Model by a list" />
</p>

  * On success, returns a list of the IDs of the stopped models.

3. **Stop All Running Models**

    Pass the string "ALL" to stop all active models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/391aa996-69e4-41c7-9c8e-56f8efaedb0c" alt="Stop DetectNet Model by a list" />
</p>
  
  * On success, returns a list of all IDs of the models that were stopped.

**Return Values**
* **ID or List of IDs** → Successful stop operation.
* `0` → No models were found that matched the given ID(s), invalid `model_id` type, or no models were running.
* `-1` → Internal error (e.g., missing required JSON keys, exception while stopping).
