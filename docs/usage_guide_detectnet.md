<h1 align = "center">Usage Guide for DetectNet </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9056f44-5639-41bb-b2b1-2473cf0680e9" alt="Jetson X Snap!" />
</p>

### Connect to the server

By using the connect to Jetson block you have to provide the IP and the port of the WebSocket server. This block returns a WebSocket Object with allow the communication betweent the server and the client. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ad79b8e-7c2c-4225-9786-6d4e846d8973" alt="Connect to Jetson Block" />
</p>


### Launch a Object Detection Model - `detectnet`

Once connected using the `connect to Jetson` block, you can launch a detection model using the `send msg to socket with response` block. This block returns a model ID `(e.g., 1000)`, which is used to reference the launched model instance.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a445f14a-448b-41a5-a789-ee0b28d9b91a" alt="Launch DetectNet Model" />
</p>

*Use the Snap! JSON extension blocks to construct the appropriate JSON request.*

#### JSON Payload Structure for detectnet /models/launch

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


### Run the Launched Object Detection Model

To perform object detection using video input, you'll need to create a loop using the repeat until block that runs continuously until a specific condition is met.

Within this loop:

1. Capture a frame from the video using the `video on` block.
2. Encode the captured image to Base64 format using the `encode base64 of` block.
3. Send the encoded image to the Jetson server using the `send base64_img to socket to model with response` block.
   
   * This block sends the image through the active WebSocket connection and using a specific detection model.
   
   * It returns a detection result in JSON format.
   
5. Display the result using the `draw detection` block, which interprets the DetectNet response and draws the detected bounding boxes on the canvas.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8194aae6-9105-4975-a5ec-d1cc84f8b2ee" alt="Run DetectNet Model" />
</p>

#### JSON Response Structure

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

### Stop the Launched Object Detection Model

It’s important to stop models after execution to free up system resources and ensure optimal performance. You can do this using the `stop model to socket` block, which supports three modes of operation depending on your needs:

1. **Stop a Single Model by ID**

    To stop a specific model, simply pass its `model_id` to the `stop model to socket` block.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c2511cb9-09a5-429e-9098-4a03806a0a89" alt="Stop DetectNet Model" />
</p>

  * The block will return the ID of the model that was successfully stopped.

2. **Stop Multiple Models Using a List**

    You can also stop multiple models at once by providing a list of model_ids.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e752b5ab-72fe-4aee-b243-6fc60ec6adb4" alt="Stop DetectNet Model by a list" />
</p>

  * The block will return a list of the IDs of all stopped models.

3. **Stop All Running Models**

    To stop all active models, pass the string "ALL" as the model ID.

<p align="center">
  <img src="https://github.com/user-attachments/assets/391aa996-69e4-41c7-9c8e-56f8efaedb0c" alt="Stop DetectNet Model by a list" />
</p>

  * This will return a list of all the model IDs that were stopped in the system.
