# NVIDIA Jetson Xavier NX (Common + PyTorch)
This is the tutorial for NVIDIA Jetson Xavier NX board.

<br>

## Device Setting
<details>
<summary>
Install Jetson Xavier NX with N100 Metal Case
</summary>
<br>
Watch this <a href="https://www.youtube.com/watch?v=7Cqr9R04htc">video guide</a>.
</details>

<details>
<summary>
Install Ubuntu OS for Jetson board
</summary>
<br>
* <a href="https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#intro">Getting Started</a> With Jetson Xavier NX Developer Kit from NVIDIA. After installation, turn on your borad and set the intial status like user account.<br>
* You can learn more <a href="https://developer.nvidia.com/jetson-xavier-nx-developer-kit-user-guide">details about the Jetson Developer Kit</a> indicating the summary of NVIDIA JetPack SDK which includes libraries and APIs such as CUDA, cuDNN, TensorRT, Multimedia API, VisionWorks, etc.
</details>

<details>
<summary>
Move boot system to SSD
</summary>
<br>
(Please follow this step, if you have installed SSD on your board.)<br>
See this <a href="https://www.youtube.com/watch?v=ZK5FYhoJqIg">video guide</a> with a <a href="https://github.com/jetsonhacks/rootOnNVMe">related github</a>.
</details>

<details>
<summary>
Configure the ip address for network connection
</summary>
<br>
For ethernet, please refer to this <a href="https://linuxize.com/post/how-to-configure-static-ip-address-on-ubuntu-18-04/">guide</a>.<br><br>
For wifi and ssh setup, see below postings:<br>
<ul>
  <li><a href="https://desertbot.io/blog/jetson-xavier-nx-headless-wifi-setup">JETSON XAVIER NX HEADLESS WIFI SETUP</a></li>
  <li><a href="https://forums.developer.nvidia.com/t/jetson-xavier-nx-dev-kit-could-not-connect-from-windows-to-nx-via-ssh/154862">Jetson Xavier NX Dev kit – could not connect from windows to NX via SSH</a></li>
  <li><a href="https://forums.developer.nvidia.com/t/unable-to-ssh-into-jetson-nano-through-ethernet/72639/8">Unable to SSH into Jetson Nano through Ethernet</a></li>
  <li><a href="https://ubuntuhandbook.org/index.php/2020/07/find-ip-address-ubuntu-20-04/">How to Find Local / Public IP Address in Ubuntu 20.04</a></li>
  <li>.ssh: connect to host HOSTNAME port 22: No route to host</li>
  <li><a href="https://unix.stackexchange.com/questions/522766/ssh-no-route-to-host">Ssh No route to host</a></li>
  <li><a href="https://www.tecmint.com/fix-no-route-to-host-ssh-error-in-linux/">How to Fix “No route to host” SSH Error in Linux</a></li>
  <li><a href="https://askubuntu.com/questions/53976/ssh-connection-error-no-route-to-host">SSH Connection Error: No route to host</a></li>
</ul>
</details>

<br>

## Basics
<details>
<summary>
Information of Jetson OS environment
</summary>
<br>
You can easily get information about the NVIDIA Jetson OS environment with this <a href="https://github.com/jetsonhacks/jetsonUtilities">github</a>.
</details>

<details>
<summary>
Power modes of Jetson board
</summary>
<br>
To display the current power mode: <code>sudo /usr/sbin/nvpmodel -q</code>.<br><br>
To change the power mode: <code>sudo /usr/sbin/nvpmodel -m [x]</code><br>
(where [x] is the power model ID, i.e. 0, 1, 2, 3, 4, 5, or 6).<br><br>
To learn about other options: <code>/user/sbin/nvpmodel -h</code>.<br><br>
** Supported Mode List **<br>
<ul>
  <li>0: MODE_15W_2CORE</li>
  <li>1: MODE_15W_4CORE</li>
  <li>2: MODE_15W_6CORE</li>
  <li>3: MODE_10W_2CORE</li>
  <li>4: MODE_10W_4CORE</li>
  <li>5: MODE_10W_DESKTOP</li>
</ul>
Here is the <a href="https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html">detailed information</a> for Jetson development.
</details>

<details>
<summary>
Saving Ram Memory with lightdm
</summary>
<br>
It may be advisable to install the lightdm desktop instead of the gdm3 when prompted during the initial installation.<br>
Please follow this <a href="https://www.jetsonhacks.com/2020/11/07/save-1gb-of-memory-use-lxde-on-your-jetson/">guide</a>.
</details>

<details>
<summary>
Using Camera
</summary>
<br>
- NVIDIA Jetson <a href="https://developer.nvidia.com/blog/jetson-xavier-nx-the-worlds-smallest-ai-supercomputer/">Xavier NX</a> board support 2 CSI lanes and up to 6 cameras with 4k video encoders and decoers.<br>
- You can install and test camera with this <a href="https://www.jetsonhacks.com/2019/04/02/jetson-nano-raspberry-pi-camera/">posting</a>.<br>
- Addtional links: <a href="https://github.com/JetsonHacksNano/CSI-Camera">link1</a>, <a href="https://github.com/NVIDIA-AI-IOT/argus_camera">link2</a>, <a href="https://forums.developer.nvidia.com/t/does-nx-encode-support-4kp60/126710/5">link3</a>.
</details>

<br>

## Environment Setting

<details>
<summary>
Install PIP for python package management
</summary>
<br>
Run the code <code>sudo apt install python3-pip</code>
</details>

<details>
<summary>
Install JTOP for interactive system monitoring
</summary>
<br>
1. Install <a href="https://github.com/rbonghi/jetson_stats">jetson-stats</a>.<br>
<code>sudo -H pip install jetson-stats</code><br><br>
2. Reboot the system.<br>
<code>sudo reboot</code><br><br>
3. Test the jtop.<br>
<code>jtop</code><br><br>
You can also check the system information at the 6th tap.<br>
To use jtop in docker container, please see <a href="https://github.com/rbonghi/jetson_stats/issues/63">issue</a>.
</details>

<details>
<summary>
NVIDIA docker for environment management
</summary>
<br>
Instead of installing packages on Jetson from scratch, you can download and run <a href="https://ngc.nvidia.com/catalog/containers?orderBy=popularDESC&pageNumber=0&query=L4T&quickFilter=&filters=">Docker containers (L4T)</a> that <a href="https://developer.nvidia.com/embedded/learn/tutorials/jetson-container">NVIDIA NGC</a> has already setup for you.<br>
For example, <code><a href="https://ngc.nvidia.com/catalog/containers/nvidia:l4t-base">NVIDIA L4T Base</a></code> enables l4t applications (kernel, necessary firmwares, NVIDIA drivers, etc) to be run in a container on Jetson.<br><br>

<b>1. Pull the container</b><br>
<code>sudo docker pull nvcr.io/nvidia/l4t-base:r32.5.0</code><br>

<b>2. Run the container</b><br>
<code>sudo docker run -it --rm --runtime nvidia --network host --name base -v /home/user/project:/location/in/container nvcr.io/nvidia/l4t-base:r32.5.0</code><br>
<code>sudo docker run -it --gpus all -v /home/nvidia/docker_yolov5/:/yolov5 -e DISPLAY=:0 -v /tmp/.X11-unix:/tmp/.X11-unix nvcr.io/nvidia/l4t-ml:r32.5.0-py3</code><br><br>
&ensp;&emsp;<i>-it</i> means run in interactive mode<br>
&ensp;&emsp;<i>--rm</i> will delete the container when finished<br>
&ensp;&emsp;<i>--runtime</i> nvidia will use the NVIDIA container runtime while running the l4t-base container<br>
&ensp;&emsp;<i><a href="https://docs.docker.com/storage/bind-mounts/">-v</a></i> is the mounting directory of the host<br>

<b>3. List the made container</b><br>
First, detach at the running container with the input sequence <code>Ctrl+P</code> followed by <code>Ctrl+Q</code>. And run <code>sudo docker ps -a</code> to see all containers on the host.<br>

There are other commands for Docker CLI such as <code>start</code>, <code>attach</code>, <code>image</code>, <code>cp</code>, etc. You can learn details <a href="https://docs.docker.com/engine/reference/commandline/docker/">here</a>.<br>

<b>For additional ML containers for Jetson</b>, see the <code><a href="https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch">NVIDIA L4T PyTorch</a></code>, <code><a href="https://ngc.nvidia.com/catalog/containers/nvidia:l4t-tensorflow">NVIDIA L4T TensorFlow</a></code>, <code><a href="https://ngc.nvidia.com/catalog/containers/nvidia:jetson-pose">Pose Demo for Jetson/L4T</a></code> images. Note that the PyTorch pip wheel installers for aarch64 used by these containers are available to download independently from the <a href="https://elinux.org/Jetson_Zoo">Jetson Zoo</a>.<br>

<b>Advaned NVIDIA-docker</b>
  <details>
  <summary>
  &emsp;Mount Plugins Specification
  </summary>
  <br>
  By default a limited set of device nodes and associated functionality is exposed within the l4t-base containers using the mount plugin capability. Internally the NVIDIA Container Runtime stack uses a plugin system to specify what files may be mounted from the host to the container. This list is documented here: <a href="https://github.com/NVIDIA/nvidia-docker/wiki/NVIDIA-Container-Runtime-on-Jetson#mount-plugins">link1, <a href="https://github.com/NVIDIA/libnvidia-container/blob/jetson/design/mount_plugins.md">link2</a>.
  </details>
</details>

<details>
<summary>
Install the recent version of OpenCV on Jetson
</summary>
<br>
<ul>
  <li>Follow this guide: <a href="https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html">Install OpenCV 4.5 on Jetson Nano</a></li>
  <li>Addtional material: <a href="https://stackoverflow.com/questions/36862589/install-opencv-in-a-docker-container">Install OpenCV in a Docker container</a></li>
</ul>
</details>

<details>
<summary>
Install PyTorch on Jetson
</summary>
<br>
<b>1. Run Docker container including PyTorch</b><br>
See the description of <code><a href="https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch">NVIDIA L4T PyTorch</a></code> image.<br><br>

<b>2. Install pip wheel from <a href="https://elinux.org/Jetson_Zoo">Jetson Zoo</a></b><br>
Follow the installation instruction of the <a href="https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048">Jetson Forum</a>.<br>

<b>3. Build PyTorch from source</b><br>
The steps are written in this <a href="https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048">thread</a>.
</details>

<br>

## Deploy YOLOv5 to Jetson Xavier NX

* References: [Link1](https://blog.roboflow.com/deploy-yolov5-to-jetson-nx/), [Link2](https://github.com/ultralytics/yolov5/issues/1944), [Link3](https://www.forecr.io/blogs/ai-algorithms/how-to-run-yolov5-real-time-object-detection-on-pytorch-with-docker-on-nvidia-jetson-modules)

### 1. Install Packages
Install OpenCV and PyTorch as in [guides](Environment-Setting) and all the other requirements like <code>pip3 install cython matplotlib pillow pyyaml scipy tensorboard tqdm seaborn pandas</code>.


### 2. Run inference on the NVIDIA Jetson NX
First, clone the repository:
```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v3.0     # if you want specific version of YOLOv5
```
Afterwards, kick off an inference session with:
```
python detect.py --source ./data/images/ --weights yolov5s.pt --conf 0.4
```
(There is an [issue](https://github.com/ultralytics/yolov5/issues/2068) that <code>detect.py</code> shows about 10 FPS for a single image while getting 30 FPS for a video, similar [issue](https://github.com/ultralytics/yolov5/issues/960) on Xavier AGX.)


### 3. Convert PyTorch to ONNX format
[ONNX](https://github.com/onnx/onnx) is an open format built to represent machine learning models from PyTorch, TensorFlow, MxNet, etc with a common file format. Internally [PyTorch's onnx submodule](https://pytorch.org/docs/1.8.1/onnx.html) supports to exchange the <code>.pt</code> format to <code>.onnx</code> format and you can run the example code <code>torch2onnx.py</code>:
```
cp torch2onnx.py yolov5/
cd yolov5/
python torch2onnx.py --weights yolov5s.pt --imgsz 640 --batch -1     # dynamic batch size
```
For static input batch, give the option <code>--batch 1</code>.

<details>
<summary>
Troubleshooting
</summary>
<br>
  <details>
  <summary>
  Each <b>opset</b> version of ONNX includes different library set
  </summary>
  <br>
  There is a dependency between this opset and TensorRT versions. For example, I used <code>opset 12</code> for <code>TensorRT=7.1.3.0</code> and <code>opset 13</code> for <code>TensorRT=8.0.1.6</code>.<br>
  
  You can check which operations are supported for each opset at the <a href="https://github.com/pytorch/pytorch/tree/master/torch/onnx">code</a> and can also add your own layers if they are not supported.
  </details>
</details>


### 4. Convert <code>.onnx</code> to <code>.engine</code> format with TensorRT
[TensorRT](https://developer.nvidia.com/tensorrt) is an SDK made by NVIDIA for optimizing trained deep learning models to enable high-performance inference, which contains a deep learning inference <b>optimizer</b> for trained deep learning models, and a <b>runtime</b> for execution. [Quick start guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#export-from-pytorch) introduces the exporting process from PyTorch to ONNX, from ONNX to a TensorRT engine, and running the engine in python. Here is [full documentation of TRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) and [installing TRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar).<br>

<code><a href="https://github.com/NVIDIA/TensorRT/blob/master/samples/opensource/trtexec/README.md">trtexec</a></code> is TensorRT command-line wrapper to quickly utilize TensorRT without having to develop your own application. For Xavier NX, This binary is located in <code>/usr/src/tensorrt/bin</code> and you can execute this to optimize the onnx model to a TRT engine.<br>
Export environment variables first:
```
export TRT_PATH="/usr/src/tensorrt"
export PATH=$PATH:$TRT_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_PATH/lib   # if you have manually installed TRT
trtexec --help    # check work well
```
Convert .pt file to .onnx with static input shapes:
```
trtexec --onnx=yolov5s_i640_b1.onnx --saveEngine=yolov5s_i640_b1.engine
# or
trtexec --onnx=yolov5s_i640_b1.onnx --saveEngine=yolov5s_i640_b1.engine --shapes=input:1x3x640x640    # to give an input shape
```
With a range of possible dynamic input shapes:
```
trtexec --onnx=yolov5s_i640_b-1.onnx --saveEngine=yolov5s_i640_b-1.engine \
--minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640 --maxShapes=input:1x3x640x640
```
For faster latency, you can give other options like <code>--fp16</code>, <code>--int8</code>, and <code>--best</code>. To see the full list of available options and their descriptions, issue the <code>trtexec --help</code> command.

<details>
<summary>
Troubleshooting
</summary>
<br>
  <details>
  <summary>
  Static model does not take explicit shapes since the shape of inference tensors will be determined by the model itself
  </summary>
  <br>
  This error occurs when you give a range of input shapes while onnx model is static. You need to set proper options as written in the above example commands.
  </details>
</details>


### 5. Run a TensorRT engine in Python
A engine can be run using [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/) or you can refer to this [github](https://github.com/RizhaoCai/PyTorch_ONNX_TensorRT).

1. [Install PyCUDA](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pycuda)
[PyCUDA](https://documen.tician.de/pycuda/index.html) is used within Python wrappers to access NVIDIA’s CUDA APIs.
```
# set environment variables
export CUDA_HOME="/usr/local/cuda-11.0"
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib

# install pycuda
pip install 'pycuda<2021.1'
```

2. Run the TRT engine
The TRT python code (<code>PythonTRT.py</code>) is based on NVIDIA's sample codes at <code>$TRT_PATH/samples/python/common.py</code> and <code>yolov3_onnx/onnx_to_tensorrt.py</code>. Run the following command to test the inference of TRT model:
```
cp PythonTRT.py yolov5/ && cp detect_trt.py yolov5/ && cd yolov5
python detect_trt.py --weights yolov5s.pt --engine yolov5s_i640_b-1.engine --imgsz 640 --source data/images/
```
You can compare the result images between <code>torch</code> and <code>tensorrt</code> at <code>yolov5/runs/detect/exp</code>. The latency comparison in various conditions are also attached in the following table.

| model | condition | device | package | latency (ms) | remarks |
| :---: | :---: | :---: | :---: | :---: | :---: |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32 | GTX 1080 Ti | PyTorch=1.8.1 | 27~29 | CUDA=11.1, cuDNN=8.0.4 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32 | GTX 1080 Ti | PyTorch=1.8.1 | 6~11 | CUDA=11.1, cuDNN=8.0.4 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32 | GTX 1080 Ti | TensorRT=8.0.1.6 | 27~28 | CUDA=11.0, cuDNN=8.0.4 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp16 | GTX 1080 Ti | TensorRT=8.0.1.6 | 27~28 | CUDA=11.0, cuDNN=8.0.4 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", int8 | GTX 1080 Ti | TensorRT=8.0.1.6 | 13~14 | CUDA=11.0, cuDNN=8.0.4 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", best | GTX 1080 Ti | TensorRT=8.0.1.6 | 13~14 | CUDA=11.0, cuDNN=8.0.4 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_DESKTOP | Jetson Xavier NX | PyTorch=1.7.0 | 699~700 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_4CORE | Jetson Xavier NX | PyTorch=1.7.0 | 484~485 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_2CORE | Jetson Xavier NX | PyTorch=1.7.0 | 484~486 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_6CORE | Jetson Xavier NX | PyTorch=1.7.0 | 401~411 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_4CORE | Jetson Xavier NX | PyTorch=1.7.0 | 401~407 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_2CORE | Jetson Xavier NX | PyTorch=1.7.0 | 399~406 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_DESKTOP | Jetson Xavier NX | PyTorch=1.7.0 | 89~90 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_4CORE | Jetson Xavier NX | PyTorch=1.7.0 | 65~66 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_2CORE | Jetson Xavier NX | PyTorch=1.7.0 | 66~67 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_6CORE | Jetson Xavier NX | PyTorch=1.7.0 | 59~62 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_4CORE | Jetson Xavier NX | PyTorch=1.7.0 | 59~61 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_2CORE | Jetson Xavier NX | PyTorch=1.7.0 | 59~62 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_DESKTOP | Jetson Xavier NX | TensorRT=7.1.3.0 | 510~514 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 338~340 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp16, MODE_10W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 109~110 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", int8, MODE_10W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 67~68 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", best, MODE_10W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 67~73 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_2CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 338~341 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_6CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 315~360 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp16, MODE_15W_6CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 94~107 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", int8, MODE_15W_6CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 52~58 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", best, MODE_15W_6CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 52~57 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 314~358 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5x | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_2CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 312~360 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_DESKTOP | Jetson Xavier NX | TensorRT=7.1.3.0 | 64~65 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 43~45 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp16, MODE_10W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 21~24 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", int8, MODE_10W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 17~18 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", best, MODE_10W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 17~18 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_10W_2CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 43~45 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_6CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 36~40 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp16, MODE_15W_6CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 17~20 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", int8, MODE_15W_6CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 13~15 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", best, MODE_15W_6CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 13~14 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_4CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 36~41 | CUDA=10.2, cuDNN=8.0.0 |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32, MODE_15W_2CORE | Jetson Xavier NX | TensorRT=7.1.3.0 | 36~40 | CUDA=10.2, cuDNN=8.0.0 |


### 6. Speeding up TensorRT
<code>PythonTRT.py</code> can be customized for different conditions such as

* Location of inputs and outputs whether CPU or GPU
  * [PyCUDA doc: DeviceAllocation](https://documen.tician.de/pycuda/driver.html#pycuda.driver.DeviceAllocation)
  * [PyCUDA doc: GPUArray](https://documen.tician.de/pycuda/array.html#pycuda.gpuarray.GPUArray)
  * [Stackoverflow: How can I create a PyCUDA GPUArray from a gpu memory address?](https://stackoverflow.com/questions/51438232/how-can-i-create-a-pycuda-gpuarray-from-a-gpu-memory-address)
* Vidieo pipline implementation
  * [Post: Speeding Up TensorRT UFF SSD](https://jkjung-avt.github.io/speed-up-trt-ssd/)
  * [(PyCUDA Context) TensorRT do_inference error](https://forums.developer.nvidia.com/t/tensorrt-do-inference-error/77055)
* Dynamic batch and image size
  * [TRT Doc: 7. Working With Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)
  * [Github Issue: how can i handle a network needs different image size](https://github.com/dusty-nv/jetson-inference/issues/192)
* INT8 calibration with custom policy
  * [TRT Doc: 5.2.3. Enabling INT8 Inference Using Python](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_python)
  * [Sample python code: int8_caffe_mnist (cache_file)](https://github.com/NVIDIA/TensorRT/tree/master/samples/python/int8_caffe_mnist)
  * [Github python: RizhaoCai/PyTorch_ONNX_TensorRT](https://github.com/RizhaoCai/PyTorch_ONNX_TensorRT)
  * [Sample c code: sampleINT8 (cache)](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleINT8#batch-files-for-calibration)
  * [Sample c code: sampleINT8API](https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleINT8API)
  * [Github c: tensorrtx/yolov5](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)
  * [Github c: enazoe/yolo-tensorrt](https://github.com/enazoe/yolo-tensorrt)
  * [Forum: How to generate int8 calilb table for trtexec engine generation](https://forums.developer.nvidia.com/t/how-to-generate-int8-calilb-table-for-trtexec-engine-generation/126015)
  * [ccoderun: sampleINT8 (batch stream)](https://www.ccoderun.ca/programming/doxygen/tensorrt/md_TensorRT_samples_opensource_sampleINT8_README.html)
  * [TRT Python API: IInt8Calibrator](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Int8/Calibrator.html)
  * [s7310-8-bit-inference-with-tensorrt.pdf](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)
  * [Forum: Int8 Calibration is not accurate](https://forums.developer.nvidia.com/t/int8-calibration-is-not-accurate-see-image-diff-with-and-without/73766)


* Infomative Documents and Githubs
  * [TensorRT Python API Reference](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)
  * [TensorRT Documentation (ccoderun)](https://www.ccoderun.ca/programming/doxygen/tensorrt/index.html)
  * [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)
  * [jkjung-avt/tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos)

Please refer to above materials to more optimizer the performance of your inference with TRT.

<br>

## Sites
- many githubs
  - https://developer.nvidia.com/embedded/community/jetson-projects
  - https://dmccreary.medium.com/getting-your-camera-working-on-the-nvida-nano-336b9ecfed3a
  - https://github.com/StrongRay/NVIDIA-Jetson-Xavier-NX
  - https://github.com/jetsonhacks/buildJetsonXavierNXKernel
  - https://github.com/dusty-nv/jetson-inference
  - https://jkjung-avt.github.io/setting-up-xavier-nx/
  - https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
- JetPack
  - https://jetpack.com/support/jetpack-cli/
  - https://docs.nvidia.com/jetson/jetpack/introduction/index.html
- Xavier NX (TensorRT)
  - https://developer.ridgerun.com/wiki/index.php?title=Xavier/Deep_Learning/TensorRT/Building_Examples#Python_API

<br>

## For more information

<details>
<summary>
NVIDIA driver is not included in the image for Jetson Nano or Xavier NX?
</summary>
<br>
Since Jetson has an integrated GPU that doesn’t use PCIe, it includes a different NVIDIA driver that comes pre-installed with JetPack on the SD card image. You don’t need to install the “nvidia-driver-430” package because that is a PCIe driver (see the <a href="https://forums.developer.nvidia.com/t/nvidia-driver-not-included-in-the-image-for-jetson-nano/76795">forum</a>).
</details>

<details>
<summary>
Max Performance Setting
</summary>
<br>
<pre><code>
$ sudo nvpmodel -m 2    # 15W 6-core mode (on Nano, use -m 0)
$ sudo jetson_clocks
</code></pre>
The <code>jetson_clocks</code> script disables the DVFS governor and locks the clocks to their maximums as defined by the active nvpmodel power mode. So if your active nvpmodel mode is 10W, <code>jetson_clocks</code> will lock the clocks to their maximums for 10W mode. You can check the source of the <code>jetson_clocks</code> shell script, and the nvpmodels are defined in <code>/etc/nvpmodel.conf</code>.
</details>

<details>
<summary>
Overclocking Jetson Nano CPU to 2 GHz and GPU to 1 GHz
</summary>
<br>
See the <a href="https://qengineering.eu/overclocking-the-jetson-nano.html">guide</a>.
</details>

<details>
<summary>
"Problem with the SSL CA cert" when running git clone
</summary>
<br>
Follow a <a href="https://thinkpro.tistory.com/148">post</a>.
</details>

<br>

## To Do List

<details>
<summary>
CUDA Installation and Managetment
</summary>
<br>
* Install CUDA 11 on Jetson: <a href="https://www.seeedstudio.com/blog/2020/07/29/install-cuda-11-on-jetson-nano-and-xavier-nx/">Link</a><br>
* Multiple Versions of CUDA on One Machine: <a href="https://medium.com/@peterjussi/multicuda-multiple-versions-of-cuda-on-one-machine-4b6ccda6faae">Link</a><br>
</details>

<details>
<summary>
How to Train YOLOv5 On a Custom Dataset
</summary>
<br>
Learn at the <a href="https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/">post</a>.
</details>

<details>
<summary>
Speeding up TensorRT
</summary>
<br>
<ul>
  <li>Using DALi for fast image or video decoding on GPU</li>
  <li>Dynamic input image size and batch size</li>
  <li>Efficient data copy between host and device with PyCUDA</li>
  <li>Build a TRT engine with python code</li>
  <li>INT8 custom calibration</li>
  <li>Profiling with NVIDIA Nsight for better scheduling</li>
</ul>
</details>