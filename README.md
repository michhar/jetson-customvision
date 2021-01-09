# Using a Custom Vision Model on an NVIDIA Jetson Device with Docker

This repo provides a sample of a [Custom Vision Service](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/) model running in a docker container on an NVIDIA Jetson device.  The resulting image will be compatible with [Live Video Analytics on IoT Edge](https://docs.microsoft.com/en-us/azure/media-services/live-video-analytics-edge/).

## Tested setup

- Jetson AGX Xavier flashed with JetPack 4.4 (L4T R32.4.3) with all ML and CV tools (**including** `nvidia-docker`)
- Additional NVMe 512 GB SSD
- 16 GB swap file (on NVMe mount if using one, otherwise main storage disk)
- [Optional] Additional NVMe 512 GB SSD
  - Set docker to use the NVMe drive for docker images
- [Optional] Azure CLI installed on Jetson to push AI docker image

If using a different Jetson device, follow instructions for flashing that specific device.  It will still be a good idea to create an 8-16 GB swap file on the available storage for the model footprint when loaded into memory.

## Instructions

1. Use CustomVision to train an object detection model, with the following notes:
  - Use "General (compact)" as "Domain" ("compact" will ensure we can export for IoT Edge)
  - Export as Dockerfile --> ARM (Raspberry Pi 3) and download the zipped folder
  - Locate the `model.pb` and `labels.txt` files in the `app` folder within the main zip folder

2. Place the `model.pb` model file and the `labels.txt` labels file into the `customvision-linux-arm/app` folder from this repo (if there is a `labels.txt` already in it, just overwrite with the newly exported one).

3. Build the docker image (this will build the image based on the `Dockerfile`) with a tag so that it's easy to upgrade as needed:
    ```
    cd customvision-linux-arm
    nvidia-docker build -t objectdetection:0.0.1 .
    ```

4. Check that it's using the GPU:
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

5. Run and test container:
    ```
    nvidia-docker run --name my_cvs_container -p 127.0.0.1:80:80 -d objectdetection:0.0.1
    curl -X POST http://127.0.0.1:80/image -F imageData=@<full path to a test image file that has the object(s)>
    ```

6. Use with Live Video Analytics on IoT Edge

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
