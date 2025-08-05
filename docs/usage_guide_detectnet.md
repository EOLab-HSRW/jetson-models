<h1 align = "center">Usage Guide for DetectNet </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9056f44-5639-41bb-b2b1-2473cf0680e9" alt="Jetson X Snap!" />
</p>

### Connect to the server

By using the connect to Jetson block you have to provide the IP and the port of the WebSocket server. This block returns a WebSocket Object with allow the communication betweent the server and the client. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ad79b8e-7c2c-4225-9786-6d4e846d8973" alt="Connect to Jetson Block" />
</p>


### Launch a detectnet model

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
