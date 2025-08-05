<h1 align = "center">Usage Guide for ImageNet </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9056f44-5639-41bb-b2b1-2473cf0680e9" alt="Jetson X Snap!" />
</p>

### Connect to the server

By using the connect to Jetson block you have to provide the IP and the port of the WebSocket server. This block returns a WebSocket Object with allow the communication betweent the server and the client. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/2ad79b8e-7c2c-4225-9786-6d4e846d8973" alt="Connect to Jetson Block" />
</p>

#### Launch a Image Recognition Model - `imagenet`

Once connected using the `connect to Jetson` block, you can launch a detection model using the `send msg to socket with response` block. This block returns a model ID `(e.g., 1000)`, which is used to reference the launched model instance.

<p align="center">
  <img src="https://github.com/user-attachments/assets/91de3a99-4f5c-4845-a4de-703e49a7c056" alt="Launch ImageNet Model" />
</p>

*Use the Snap! JSON extension blocks to construct the appropriate JSON request.*

#### JSON Payload Structure for imagenet /models/launch

The `send msg to socket with response` block expects a JSON object like this:

```json
{
  "command": "/models/launch",
  "model_name": "imagenet",
  "variant_name": "googlenet",
  "topK": 1
}
```

#### JSON Argument Reference

| Key          | Type    | Required  | Default   |
|:-------------|:--------|:----------|:----------|
| command      | string  | Yes       | —         |
| model_name   | string  | Yes       | —         |
| variant_name | string  | No        | googlenet |
| topK         | Integer | No        | 1         |

*You can omit optional fields if you're happy with the defaults. However, specifying them gives more control.*

#### Available Image Recognition Variants

| Variant             | variant_name Argument   |
|:--------------------|:------------------------|
| AlexNet             | alexnet                 |
| GoogleNet (Default) | googlenet	              | 
| GoogleNet-12        | googlenet-12            |
| ResNet-18           | resnet-18               |
| ResNet-50           | resnet-50               |
| ResNet-101          | resnet-101	            | 
| ResNet-152          | resnet-152              |
| VGG-16              | vgg-16	                |
| VGG-19              | vgg-19		              | 
| Inception-v4        | inception-v4            |

