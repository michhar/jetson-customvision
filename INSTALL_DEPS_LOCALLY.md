# Install Python Dependencies for Local Testing

## Prerequisites

A Python3 virtual environment is recommended.  To install the Python `venv` tool, run:

```
sudo apt-get install python3-venv
```

To setup the virtual environment (sets up a folder called `env`), run:

```
python3 -m venv env
```

Activate the virtual environment with:

```
source env/bin/activate
```

Now, packages may be installed into this nuclear environment so as not to complicate the base Python install (and remove easily as needed).

## Install ONNX Runtime

1.  [Build ONNX on ARM64](https://github.com/onnx/onnx#build-onnx-on-arm-64)
2.  [Install ONNX Runtime (1.6.0)](https://elinux.org/Jetson_Zoo#ONNX_Runtime)

## Install PIL/Pillow

First, install the dependencies:

```
sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev libfribidi-dev tcl8.6-dev tk8.6-dev python3-tk
```

Then, pip install Pillow:

```
pip install Pillow
```
