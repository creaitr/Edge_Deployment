#  Pixel 3
Deployment on Pixel 3 XL

## Spec
- CPU: Qualcomm SDM845 Snapdragon 845 (10 nm)
- GPU: Adreno 630
- more details: [url1](https://www.gsmarena.com/google_pixel_3_xl-9257.php), [url2](https://en.wikipedia.org/wiki/Pixel_3)

## Index
1. [PyTorch](#PyTorch)
2. [TensorFlow](#TensorFlow) (Not yet)
3. [ONNX runtime](#ONNX-runtime) (Not yet)

## PyTorch

#### Prerequisites
- Setting PyTorch ([Guide](../../frameworks/PyTorch/setting.md))
- Setting Android Studio ([Guide](common.md))

#### Contents

1. [Quickstart with a HelloWorld example](../../frameworks/PyTorch/Android.md#Quickstart-with-a-HelloWorld-example)
2. [Use of camera output](../../frameworks/PyTorch/Android.md#Use-of-camera-output)
3. [Object detection and other tasks](../../frameworks/PyTorch/Android.md#Object-detection-and-other-tasks)
4. [Optimization of performance on mobile](../../frameworks/PyTorch/Android.md#Optimization-of-performance-on-mobile)

#### In progress
All above tutorials are using only CPUs. If you want to use GPU, TPU, or others, please see the following descriptions:

- [PyTorch with Vulkan](../../frameworks/PyTorch/Android.md#PyTorch-with-Vulkan-backend-for-GPU)
- [PyTorch with NNAPI](../../frameworks/PyTorch/Android.md#PyTorch-with-NNAPI)


## To Do List

<details>
<summary>
TensorFlow
</summary>
<br>
  - env<br>
  - tf==2.4.<br>
  - CUDA 10.2<br>
1. make model<br>
2. use tf-lite
</details>

<details>
<summary>
ONNX runtime
</summary>
<br>
  - env<br>
  - onnx==1.8.0<br>
  - onnxruntime==0.5.0<br>
1. tf, torch -> onnx<br>
2. onnx -> tf, torch<br>
3. use onnxruntime<br>
<a href="https://cloudblogs.microsoft.com/opensource/2020/10/12/introducing-onnx-runtime-mobile-reduced-size-high-performance-package-edge-devices/">Introduction of ONNX Runtime</a>
</details>
