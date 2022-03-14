# PyTorch Preparation
From environment settings of PyTorch to load and run of pre-built models.


## Prerequisites
- Getting started with [conda](../conda.md)


## Contents
* [Set a conda environment for PyTorch](#Set-a-conda-environment-for-PyTorch)
* [Run pre-built PyTorch models](#Run-pre-built-PyTorch-models)
    * Torchvision models
    * PyTorch Hub

</br>

## Set a conda environment for PyTorch
Install PyTorch and related packages with a conda environment.

### 1. Find proper package versions.
1. Check NVIDIA Driver Version.
    ```
    $ nvidia-smi
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 440.100      Driver Version: 440.100      CUDA Version: 10.2     |
    |-------------------------------+----------------------+----------------------+
    ```
    where, CUDA Version is the maximum available version regarding to Driver version. You can also check proper cuDNN version according to GPU architecture and CUDA version at [cudnn-support-matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html).

2. By the following command, you can search and determine proper versions (e.g., python, CUDA, cuDNN, torch, torchvision, etc).
    ```
    conda search torch -c pytorch
    ```
    
3. Here is a version setting example:
    - NVIDIA Driver: 440.100
    - CUDA: 10.2
    - cuDNN: 7.6.5
    - Python: 3.8.8
    - torch==1.8.1
    - torchvision==0.9.1

### 2. Create a conda environment and install packages.
There are two ways, use of only command lines or a already configured yml file.

- Only command lines (refer to [pytorch.org](https://pytorch.org/)).
    ```
    conda create -n [name] python=[3.8]
    conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
    ```
    Installation instructions and binaries for previous PyTorch versions may be found on [this website](https://pytorch.org/get-started/previous-versions/).

- Use a yml file (an example [yml file](environment.yml)).
    
    Make a yml file
    ```
    name: jb
    channels:
        - pytorch
        - defaults
    dependencies:
        # ...
        - cudatoolkit=10.2.89=hfd86e86_1
        # ...
        - pip=21.0.1=py38h06a4308_0
        - python=3.8.8=hdb3f193_5
        - pytorch=1.8.1=py3.8_cuda10.2_cudnn7.6.5_0
        # ...
        - torchaudio=0.8.1=py38
        - torchvision=0.9.1=py38_cu102
    ```
    and run the below command.
    ```
    conda env create -n [name] --file environment.yml
    ```
    (You can check more detailed [management of packages](https://towardsdatascience.com/managing-cuda-dependencies-with-conda-89c5d817e7e1).)

### 3. Activate the created environment.
Activate the environment
```
conda activate [name]
```
and check installed PyTorch.
```
python
import torch
print(toch.__version__)
```

<br>

## Run pre-built PyTorch models

### Torchvision models

1. There are various models in [torchvision.model](https://pytorch.org/vision/stable/models.html) subpackage for different vision tasks.
    - Classificiation: AlexNet, VGG, ResNet, ShuffleNet v2, MobileNet v2 and v3, ResNeXt, MNASNet, etc.
    - Object Detection: Faster R-CNN, Mask R-CNN, RetinaNet
    - Semantic Segmentation: FCN, DeepLabV3, LR-ASPP

2. A python script example for classficiation models.

    ```
    import torch
    import torchvision

    name = 'mobilenet_v3_small'

    # load a pre-trained model
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'mobilenet_v2':
        model = models.quantization.mobilenet_v2(pretrained=True, quantize=False)
    elif name == 'mobilenet_v3_small':
        model = torchvision.models.mobilenet_v3_small(pretrained=True, quantize=False)
    model.eval()

    # test inference
    example = torch.rand(1, 3, 224, 224)
    out = model(example)
    ```
    
    More descriptions are [here](https://pytorch.org/vision/stable/models.html). There are other libraries such as torchvideo, torchaudio, torchtext, etc.


### PyTorch Hub

[PyTorch Hub](https://pytorch.org/docs/stable/hub.html) is a pre-trained model repository designed to facilitate research reproducibility (a [guide](https://jybaek.tistory.com/813) of PyTorch Hub written in Korean).

1. See well-known models in PyTorch Hub.

    - Classificiation: [ResNet](https://pytorch.org/hub/pytorch_vision_resnet/), [MobileNet v2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
    - Object Detection: [SSD](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/), [YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/)
    - Semantic Segmentation: [FCN-ResNet101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/), [Deeplabv3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)

    There are more pubilshed [models](https://pytorch.org/hub/#model-row).

2. Python script example for classficiation models.

    ```
    import torch
    import torchvision

    name = 'mobilenet_v2'
    
    # load a pre-trained model
    if name == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    elif name == 'mobilenet_v2':
        model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    # test inference
    example = torch.rand(1, 3, 224, 224)
    out = model(example)
    ```
