# Smart Conveyor Belt for Honey Jar Detection and Sorting

This project implements a smart conveyor belt system that utilizes YOLOv5 object detection and ESP32 microcontrollers for detecting the conditions of honey jars and sorting them accordingly. The entire system is designed to operate wirelessly, with each device connected to a local Wi-Fi network.

The project involves three main components: a main computer (laptop), an ESP32 microcontroller for camera feed, and another ESP32 microcontroller for sorting actions. The main computer is responsible for honey jar detection using YOLOv5 as the object detection model. Image processing techniques are applied to detect the liquid level inside each jar. The first ESP32 (ESP32-CAM) captures the camera feed, which is then streamed in real-time using the Real-Time Streaming Protocol (RTSP) to the main computer for object detection. The detected liquid level determines the appropriate action to be taken on each jar. Additionally, the condition of the cap and label are also monitored, triggering respective actions for defective jars.

The second ESP32 microcontroller is responsible for sorting the jars based on their conditions. Socket programming is employed for communication between the Python code running on the main computer and the second ESP32 microcontroller over the local Wi-Fi network. Messages are sent from the Python code to the second ESP32 whenever an anomaly is detected in a honey jar. The second ESP32 microcontroller then takes the necessary action on the respective jar using a pneumatic sorting mechanism.

## Features

- Real-time detection of honey jar conditions using YOLOv5 object detection.
- Image processing for liquid level detection inside the jars.
- Monitoring and action-taking based on cap and label conditions.
- ESP32-CAM for capturing and streaming camera feed to the main computer.
- Wireless communication between devices via a local Wi-Fi network.
- Socket programming for communication between Python code and the second ESP32 microcontroller.
- Pneumatic system for sorting the jars based on their conditions.

## Prerequisites

To run this project, the following prerequisites are required:

- Main computer (laptop) with the necessary software and libraries for running the Python code.
- ESP32 microcontrollers (ESP32-CAM and the second ESP32) with the required firmware and software.
- Wi-Fi network for connecting all devices.

## Installation

1. Clone the repository: `https://github.com/PranayLendave/industry4.0-with-yolov5-and-esp32.git`
2. Install the necessary dependencies and libraries as specified in the `requirements.txt` file.
3. Configure the ESP32-CAM and the second ESP32 microcontroller according to the provided instructions.
4. Set up the Wi-Fi network and ensure all devices are connected to it.

## Usage

1. Run the Python code on the main computer to initiate the honey jar detection system.
2. The file `main.c` contains the main program running the project. 
3. Ensure the ESP32-CAM is streaming the camera feed to the main computer via RTSP.
4. Monitor the detected conditions of the honey jars on the main computer.
5. If any anomaly is detected, the Python code will send messages to the second ESP32 microcontroller.
6. The second ESP32 microcontroller will perform the necessary sorting action using the pneumatic system.

## Contributing

Major contributors:
- Pranay Lendave
- Viral Faria 

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please submit bug reports or feature requests through the issue tracker.

