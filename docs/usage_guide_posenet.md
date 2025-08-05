<h1 align = "center">Usage Guide for PoseNet </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9056f44-5639-41bb-b2b1-2473cf0680e9" alt="Jetson X Snap!" />
</p>

### Connect to the server

By using the connect to Jetson block you have to provide the IP and the port of the WebSocket server. This block returns a WebSocket Object with allow the communication betweent the server and the client. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ad79b8e-7c2c-4225-9786-6d4e846d8973" alt="Connect to Jetson Block" />
</p>

### Launch a Pose Estimation Model - `posenet`

Once connected using the `connect to Jetson` block, you can launch a detection model using the `send msg to socket with response` block. This block returns a model ID `(e.g., 1000)`, which is used to reference the launched model instance.

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
