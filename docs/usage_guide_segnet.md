<h1 align = "center">Usage Guide for DetectNet </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9056f44-5639-41bb-b2b1-2473cf0680e9" alt="Jetson X Snap!" />
</p>

### Connect to the server

By using the connect to Jetson block you have to provide the IP and the port of the WebSocket server. This block returns a WebSocket Object with allow the communication betweent the server and the client. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ad79b8e-7c2c-4225-9786-6d4e846d8973" alt="Connect to Jetson Block" />
</p>

### Launch a Segmentation Model - `segnet`

Once connected using the `connect to Jetson` block, you can launch a detection model using the `send msg to socket with response` block. This block returns a model ID `(e.g., 1000)`, which is used to reference the launched model instance.

<p align="center">
  <img src="https://github.com/user-attachments/assets/33f21e3b-b55e-4f40-a1f2-3796860bbb85" alt="Launch SegNet Model" />
</p>

*Use the Snap! JSON extension blocks to construct the appropriate JSON request.*

#### JSON Payload Structure for segnet /models/launch

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

#### JSON Argument Reference

| Key          | Type    | Required  | Default                  |
|:-------------|:--------|:----------|:-------------------------|
| command      | string  | Yes       | —                        |
| model_name   | string  | Yes       | —                        |
| variant_name | string  | No        | fcn-resnet18-voc-320x320 |
| filter_mode  | string  | No        | linear                   |
| alpha        | float   | No        | 150.0                    |
| ignore_class | string  | No        | void                     |
| visualize    | string  | No        | overlay,mask             |

*You can omit optional fields if you're happy with the defaults. However, specifying them gives more control.*

#### Available Segmentation Variants

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
