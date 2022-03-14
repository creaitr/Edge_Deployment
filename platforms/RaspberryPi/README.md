# Raspberry Pi 4 (Common + PyTorch)
This is the tutorial for Raspberry Pi 4 \[<a href="https://www.raspberrypi.org/products/raspberry-pi-4-model-b/specifications/">specs</a>\].

<br>

## Device Setting
<details>
<summary>
Setting Ubuntu Desktop OS on Raspberry Pi 4
</summary>
<br>
First of all, download the image file at <a href="https://ubuntu.com/download/raspberry-pi">here</a>.<br>
And follow the tutorial <a href="https://ubuntu.com/tutorials/how-to-install-ubuntu-desktop-on-raspberry-pi-4#1-overview">how to install Ubuntu Desktop on Raspberry Pi 4</a>.
</details>

<details>
<summary>
Configure the ip address for network connection
</summary>
<br>
<b>To configure a static ip address (<a href="https://linuxize.com/post/how-to-configure-static-ip-address-on-ubuntu-18-04/">link</a>):</b><br>
1. Install a package <code>sudo apt install net-tools</code> and test the <code>ifconfig</code> command.<br>
2. Change the file <code>sudo nano /etc/netplan/01-network-manager-all.yaml</code> with the below example.<br>
<pre><code>
network:
  version: 2
  renderer: networkd
  ethernets:
    ens3:
      dhcp4: no
      addresses:
        - 192.168.121.199/24
      gateway4: 192.168.121.1
      nameservers:
          addresses: [8.8.8.8, 1.1.1.1]
</code></pre>
3. Apply the change with <code>sudo netplan apply</code>.<br>
4. Verify the change with <code>ip addr show dev eth0</code>.<br>
<br>
<b>To configure the ssh connection:</b><br>
1. Install a package <code>sudo apt install openssh-server</code>.<br>
<br>
(optional - change ssh port)<br>
2. Open the <code>/etc/ssh/sshd_config</code> file and locate the line <code>#Port 22</code>.<br>
3. Then, uncomment it and change the value with an appropriate number, e.g., 22000, <code>Port 22000</code>.<br>
4. Restart the SSH server with <code>systemctl restart sshd</code>.<br>
5. Verify the change with <code>netstat -tulpn | grep ssh</code>.<br>
<br>
(optional - set up a firewall with UFW)<br>
6. Install a package <code>sudo apt install ufw</code>.<br>
7. Add 'ssh-server' and the port number to the allow list.<br>
<pre><code>
sudo ufw allow openssh
sudo ufw allow 22000/tcp
</code></pre>
8. Enable the change with <code>sudo ufw enable</code> or <code>sudo ufw reload</code>.<br>
9. Verify the change with <code>sudo ufw status</code>.<br>
<br>
<b>Finally</b>, you can try the ssh connection like <code>ssh -p 22000 user@127.0.0.1</code>.
</details>

<br>

## Environment Setting

<details>
<summary>
Change the version of python on Raspberry Pi
</summary>
<br>
For Ubuntu desktop 21.04 OS, Python 3.9.5 is default.<br>
However, you can install other versions of python such as 3.8 as in this <a href="https://installvirtual.com/how-to-install-python-3-8-on-raspberry-pi-raspbian/">reference</a>.<br>
If you further need to change the link for the new python, please refer to <a href="https://codechacha.com/ko/change-python-version/">this</a>.
</details>

<details>
<summary>
Install PyTorch library on a Raspberry Pi 4
</summary>
<br>
You can simply follow this <a href="https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html">guide</a>.
</details>

<details>
<summary>
Install OpenCV 4.5 on Raspberry 64 OS
</summary>
<br>
Follow this <a href="https://qengineering.eu/install-opencv-4.5-on-raspberry-64-os.html">instruction</a>.<br>
<br>

  <details>
  <summary>
  Troubleshooting
  </summary>
  <br>
    <details>
    <summary>
    There are no <code>/usr/bin/zram.sh</code> or <code>/sbin/dphys-swapfile</code> files in the system (Ubuntu 21.04)
    </summary>
    <br>
    Please check this <a href="https://linuxconcept.com/how-to-add-swap-space-on-ubuntu-21-04-operating-system/">guide</a> or the followings:<br>
    <b>1. Increase the swap space before building OpenCV</b>
    <pre><code>
    sudo swapoff -a
    sudo fallocate -l 3G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    sudo swapon --show
    sudo free -h
    </code></pre>
    <b>2. Reset the swap space as the previous setting after the installation of OpenCV</b><br>
    Disable the swap space with <code>sudo swapoff -a</code>.<br>
    Edit the <code>/etc/fstab</code> file and remove the line <code>/swapfile swap swap defaults 0 0</code> from the file and save.<br>
    After the remove of the swap file, you can again follow the first step to reset the swap file with 1G.
    </details>
  </details>
</details>

<br>

## Deploy YOLOv5 to a Raspberry Pi 4
- Reference: [The tutorial of Jetson Xavier NX](../Jetson/XavierNX/README.md#deploy-yolov5-to-jetson-xavier-nx)

### 1. Install Packages
Install OpenCV and PyTorch as in [guides](Environment-Setting) and all the other requirements like <code>pip install cython matplotlib pillow pyyaml scipy tensorboard tqdm seaborn pandas</code>.

<details>
<summary>
Troubleshooting
</summary>
<br>
  <details>
  <summary>
  subprocess.CalledProcessError: Command '('lsb_release', '-a')' returned non-zero exit status 1.
  </summary>
  <br>
  You can refer to this <a href="https://github.com/pypa/pip/issues/4924">thread</a> or<br>
  simply run the command <code>sudo mv /usr/bin/lsb_release /usr/bin/lsb_release_back</code>.
  </details>
</details>

### 2. Run inference on the Raspberry Pi 4
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

The below table denotes the latency of yolo models on the Raspberry Pi 4 board.

| model | condition | device | package | latency (ms) | remarks |
| :---: | :---: | :---: | :---: | :---: | :---: |
| yolov5s | img(640), batch(1), "data/images/bus.jpg", fp32 | Raspberry Pi 4 (4GB) | PyTorch=1.8.0 | 1401 | cpu |

<br>

## To Do List

<details>
<summary>
Deep learning examples on Raspberry 32/64 OS
</summary>
<br>
There are many <a href="https://qengineering.eu/deep-learning-examples-on-raspberry-32-64-os.html">tutorials</a>, e.g., object detection, segmentation, face detection, super-resolution, etc.
</details>

<details>
<summary>
<b>Backend configuration for mobile environments</b>
</summary>
<br>
There are three options:<br>
1. <a href="https://github.com/Maratyszcza/NNPACK">NNPACK</a><br>
2. <a href="https://github.com/google/XNNPACK">XNNPACK</a><br>
3. <a href="https://github.com/pytorch/QNNPACK">QNNPACK</a><br>
<br>
For configuration recipe, please refer to <a href="https://discuss.pytorch.org/t/controlling-whether-pytorch-uses-nnpack-or-thnn/65852">link1</a> and <a href="https://github.com/pytorch/pytorch/issues/30622">link2</a>.
</details>

<details>
<summary>
SF flashcard life issue
</summary>
<br>
https://qengineering.eu/protect-the-raspberry-pi-4-sd-flashcard.html
</details>

<details>
<summary>
ARMnn
</summary>
<br>
https://qengineering.eu/install-armnn-on-raspberry-pi-4.html
</details>

<details>
<summary>
TBA
</summary>
<br>
.
</details>