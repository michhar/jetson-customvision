# Using a Custom Vision Model on an NVIDIA Jetson Device with Docker

This repo provides a sample of a [Custom Vision Service](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/) model running in a docker container on an NVIDIA Jetson device.  It utilizes the NVIDIA GPU for predictions.  The resulting docker image will be compatible with [Live Video Analytics on IoT Edge](https://docs.microsoft.com/en-us/azure/media-services/live-video-analytics-edge/).

## Tested setups

Jetson AGX Xavier

- Jetson AGX Xavier flashed with JetPack 4.4 (L4T R32.4.3) with all ML and CV tools (**including** `nvidia-docker`)
- 8-16 GB swap file (on NVMe mount if using one, otherwise main storage disk) for expanding memory during prediction
- [Optional - for expanded storage] Additional NVMe 512 GB SSD
  - Set docker to use the NVMe drive for docker images
  
Jetson Nano

- Jetson Nano flashed with Jetpack 4.4.1 (L4T R32.4.4) using standard setup path, [Getting Started with Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
- Using 128 GB SD card for OS and system
- Set up additional swap (8GB total shared w/ CPUs) with [instructions from Jetson Hacks](https://www.jetsonhacks.com/2019/11/28/jetson-nano-even-more-swap/)
- [Optional - for faster read/writes and expanded storage] Additional SSD mounted through USB 3.0 port ([instructions on Jetson Hacks](https://www.jetsonhacks.com/2019/09/17/jetson-nano-run-from-usb-drive/))
  - Set docker to use the SSD drive for docker images

Software on devices

- OS:  Ubuntu 18.04 LTS
- [Optional] Azure CLI installed on Jetson to push AI docker image
- Default docker (not Moby) or Docker CE

Notes

- Follow instructions for flashing that specific device (e.g. Jetson Nano using a prebuilt image for a SD-card based OS from NVIDIA link above)
- Building and running the docker images will be with `nvidia-docker`
- Memory on Nano is shared between CPUs and GPU so increasing the swap size will help greatly

To create a swap file, follow these Linux instructions:  https://linuxize.com/post/how-to-add-swap-space-on-ubuntu-18-04/#creating-a-swap-file.

## Instructions

### 1. Use CustomVision to train an object detection model

Notes

  - Use **"General (compact)"** as "Domain" ("compact" will ensure we can export for IoT Edge and model is of a smaller architecture)
  - If there are multiple classes, ensure a balanced dataset, that is, same number of images in each class for best performance ([other tips from Microsoft](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/getting-started-improving-your-classifier))
  - Export as Dockerfile --> ARM (Raspberry Pi 3) and download the zipped folder
  - Locate the `model.pb` and `labels.txt` files in the `app` folder within the main zip folder

### 2. Add model and labels to project

- Place the `model.pb` model file and the `labels.txt` labels file into the `customvision-linux-arm/app` folder from this repo (if there is a `labels.txt` already in it, just overwrite with the newly exported one).

### 3. Build the docker image

Note:  this will build the image based on the `Dockerfile`.  Use a tag that is **not "latest"** so that it's easy to upgrade as needed (e.g. shown here as `0.0.1`).

```
cd customvision-linux-arm
nvidia-docker build -t objectdetection:0.0.1 .
```

### 4. Check that it's using the GPU

Here, run the container and log into it.

```
nvidia-docker run -it objectdetection:0.0.1 /bin/bash

# Inside container start Python 3
eroot@:/app# python

# Inside Python interpreter
Python 3.6.9 (default, Oct  8 2020, 12:12:24) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# You should see more than 0 GPUs
>>> exit() # exit out of Python interpreter

# Try to run the Python app
eroot@:/app# python -u app.py

# Stop this container
eroot@:/app# exit
```

### 5. Test container predictions

Use an image file similar to your training dataset.

```
nvidia-docker run --name my_cvs_container -p 127.0.0.1:80:80 -d objectdetection:0.0.1
curl -X POST http://127.0.0.1:80/image -F imageData=@<full path to a test image file that has the object(s)>
```

The results will look like:

```
{"created":"2021-01-30T00:30:42.895140","id":"","iteration":"","predictions":[{"boundingBox":{"height":0.12187065,"left":0.76290076,"top":0.32214137,"width":0.05968696},"probability":0.78185642,"tagId":0,"tagName":"no_hardhat"}],"project":""}
```

Note
- The position indicated by "left" and "top" refers to the distances from the top-left corner of the image

### 6. Use with Live Video Analytics on IoT Edge

  - This docker image can now be [pushed to Azure Container Registry with the Azure CLI](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli) and used with Live Video Analytics on IoT Edge on a Jetson device registered with Azure IoT Hub

- [Set up the IoT Edge runtime on Jetson device](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge?view=iotedge-2018-06&tabs=linux)
  - Ensure on IoT Edge 1.0.10.x

- Ensure the "cv" ML IoT module has the following `createOptions` (see the included `deployment.customvision.arm64.template.json` deployment manifest):

    ```
    "createOptions": {
      "HostConfig":{"runtime": "nvidia",
      "Privileged":true}
    }
    ```

- Ensure the docker daemon config is using the NVIDIA runtime.  This config may be found at `/etc/docker/daemon.json` and will look like (or similar to):

    ```
    {
        "runtimes": {
            "nvidia": {
                "path": "nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
    ```
### Troubleshooting

- **Check that the GPU is being utlized**

To check with a package called `jetson-stats`, Python3 is needed (on Jetson, `nvidia-smi` is not available).  Install Python as follows.

```
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libffi-dev \
    libssl1.0.0 \
    libssl-dev \
    python3-dev \
    python3-pip
```

Install `jetson-stats` (more here):

```
sudo pip3 install jetson-stats
```

Run the tool to check resource management and usage:

```
jtop
```

- **If the device is sluggish or appears frozen**

To work with a device that is sluggish, try to SSH in with the command line from a host computer rather than with a USB-attached keyboard and mouse and HDMI attached monitor (turn off device and dettach all of these components, then power back on).  This will usually allow more responsiveness.  If there is a module that is causing issues or direct method that can not be deactivated, the IoT Edge runtime may need to be uninstalled and reinstalled.  Refer to some of the instructions in [Uninstall IoT Edge](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge?view=iotedge-2018-06&tabs=linux#uninstall-iot-edge).

If the device is entirely frozen even after using an SSH method rather than monitor, keyboard and mouse, the device may need to be reflashed.

- **If prediction is unacceptably slow**

To speed up the time for prediction, adjust the input resolution on line 18 of `~/Documents/jetson-customvision/customvision-linux-arm\object_detection.py` (https://github.com/michhar/jetson-customvision/blob/main/customvision-linux-arm/app/object_detection.py#L18) to be smaller, e.g., `256 * 256`.
