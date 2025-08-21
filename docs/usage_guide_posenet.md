<h1 align = "center">PoseNet Usage Guide with Snap!</h1>

<p align="center">
  <img src="/docs/images/posenet_demostration.gif" alt="PoseNet demonstration" height="70%" width="70%" />
</p>

## Connect to the server

By using the `connect to Jetson` block you have to provide the IP and the port of the WebSocket server. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ad79b8e-7c2c-4225-9786-6d4e846d8973" alt="Connect to Jetson Block" />
</p>

**Return values**
* This block returns a **`WebSocket Object`** that allow the communication betweent the server and the client.

## Launch a Pose Estimation Model - `posenet`

Once connected using the `connect to Jetson` block, you can launch a detection model using the `send msg to socket with response` block. This block returns a model ID `(e.g., 1000)`, which is used to reference the launched model instance.

<p align="center">
  <img src="https://github.com/user-attachments/assets/38ffff35-7612-45b2-925a-baff2f6cb697" alt="Launch PoseNet Model" />
</p>

*Use the Snap! JSON extension blocks to construct the appropriate JSON request.*

### JSON Payload Structure for posenet /models/launch

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

### JSON Argument Reference

| Key          | Type    | Required  | Default       |
|:-------------|:--------|:----------|:--------------|
| command      | string  | Yes       | —             |
| model_name   | string  | Yes       | —             |
| variant_name | string  | No        | resnet18-body |
| overlay      | string  | No        | none          |
| threshold    | float   | No        | 0.15          |

*You can omit the fields that are not required if you're happy with the defaults. However, specifying them gives more control.*

### Available Pose Estimation Variants

| Variant                       | variant_name Argument   |
|:------------------------------|:------------------------|
| Pose-ResNet18-Body (Default)  | resnet18-body           |
| Pose-ResNet18-Hand            | resnet18-hand	          | 
| Pose-DenseNet121-Body         | densenet121-body        |


## Run the Launched Pose Estimation Model

To perform pose estimation using video input, you'll need to create a loop using the `repeat until` block that runs continuously until a specific condition is met.

Within this loop:

1. Capture a frame from the video using the `video on` block. 
2. Encode the captured image to Base64 format using the `encode base64 of` block.

   **Return Values**
   * **Base64 string** → Successful encoding.
   * `-1` → Error: input was not a valid image.
     
4. Send the encoded image to the Jetson server using the `send base64_img to socket to model with response` block. This block sends the image through the active WebSocket connection executing using a specific posenet model.

   **Return Values**
   * **List of poses (JSON objects)** → Successful inference.
   * **Empty list `[]`** → One or more parameters were invalid (e.g., wrong model ID, invalid image, missing connection).
   
6. Display the result using the `draw poses` block. This block renders the points and links connections on top of the image using poses list.

   **Behavior**
   * If the poses list is **non-empty**, renders the points and links connections.
   * If the poses list is **empty**, the block does nothing (no error, just skipped).

<p align="center">
  <img src="https://github.com/user-attachments/assets/64bdf55d-8bd8-4963-a730-fee1094f7c88" alt="Run PoseNet Model" />
</p>

### JSON Response Structure

Here’s an example of the response returned from a successful inference using posenet:

```json
[
    {
        "Poses": {
            "Keypoints": [
                {
                    "ID": 0,
                    "Name": "nose",
                    "x": 0.4542,
                    "y": 0.6303
                },
                {
                    "ID": 1,
                    "Name": "left_eye",
                    "x": 0.5024,
                    "y": 0.5596
                },
                {
                    "ID": 2,
                    "Name": "right_eye",
                    "x": 0.4123,
                    "y": 0.5708
                },
                {
                    "ID": 3,
                    "Name": "left_ear",
                    "x": 0.5894,
                    "y": 0.5797
                },
                {
                    "ID": 4,
                    "Name": "right_ear",
                    "x": 0.3632,
                    "y": 0.5872
                },
                {
                    "ID": 5,
                    "Name": "left_shoulder",
                    "x": 0.6928,
                    "y": 0.8672
                },
                {
                    "ID": 6,
                    "Name": "right_shoulder",
                    "x": 0.2728,
                    "y": 0.9042
                },
                {
                    "ID": 17,
                    "Name": "neck",
                    "x": 0.4908,
                    "y": 0.8922
                }
            ],
            "Links": [
                {
                    "ID1": 1,
                    "ID2": 2
                },
                {
                    "ID1": 0,
                    "ID2": 1
                },
                {
                    "ID1": 0,
                    "ID2": 2
                },
                {
                    "ID1": 1,
                    "ID2": 3
                },
                {
                    "ID1": 2,
                    "ID2": 4
                },
                {
                    "ID1": 3,
                    "ID2": 5
                },
                {
                    "ID1": 4,
                    "ID2": 6
                },
                {
                    "ID1": 0,
                    "ID2": 7
                },
                {
                    "ID1": 5,
                    "ID2": 7
                },
                {
                    "ID1": 6,
                    "ID2": 7
                }
            ]
        }
    }
]
```

## Stop the Launched Pose Estimation Model

It’s important to stop models after execution to free up system resources and ensure optimal performance. You can do this using the `stop model to socket` block, which supports three modes of operation depending on your needs:

1. **Stop a Single Model by ID**

    Provide the specific `model_id` to stop a single model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7fa38907-431d-457f-92c9-d266493d0d07" alt="Stop PoseNet Model" />
</p>

 * On success, returns the ID of the stopped model.

2. **Stop Multiple Models Using a List**

    Provide a list of `model_ids` (e.g., `[1000,1002,1025]`).

<p align="center">
  <img src="https://github.com/user-attachments/assets/e752b5ab-72fe-4aee-b243-6fc60ec6adb4" alt="Stop DetectNet Model by a list" />
</p>

  * On success, returns a list of the IDs of the stopped models.

3. **Stop All Running Models**

    Pass the string `"ALL"` to stop all active models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/391aa996-69e4-41c7-9c8e-56f8efaedb0c" alt="Stop DetectNet Model by a list" />
</p>
  
  * On success, returns a list of all IDs of the models that were stopped.

**Return Values**
* **ID or List of IDs** → Successful stop operation.
* `0` → No models were found that matched the given ID(s), invalid `model_id` type, or no models were running.
* `-1` → Internal error (e.g., missing required JSON keys, exception while stopping).



