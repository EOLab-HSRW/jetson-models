<h1 align = "center">Jetson Model Manager X Snap! </h1>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c9056f44-5639-41bb-b2b1-2473cf0680e9" alt="Jetson X Snap!" />
</p>

This repository provides a Python-based models manager designed for running AI inference models on a Jetson device. It leverages NVIDIA's Jetson Inference Library to manage the models. The manager also sets up a WebSocket server, enabling remote clients to control different models and send images for real-time inference.

While the primary goal of this project is to facilitate model interaction through Snap!, a block-based programming language ideal for educational and demonstrative purposes, it also supports connectivity from any programming language capable of WebSocket communication. This flexibility allows users to integrate the model manager into diverse environments, as long as the client supports WebSocket connections. Whether through Snap! or your preferred language, this manager simplifies the launch, retraining, and use of AI detection models on Jetson devices.
