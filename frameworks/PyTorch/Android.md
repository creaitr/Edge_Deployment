Deployment with PyTorch Mobile
===
Basic tutorial of deployment of PyTorch models on Android OS using [PyTorch Mobile](https://pytorch.org/mobile/home/). This is based on this [examples](https://github.com/pytorch/android-demo-app).


## Prerequisites
- User Guide of [Android Studio](../../platforms/Android/common.md#build-and-run-with-android-studio).


## Contents
* [Quickstart with a HelloWorld example](#Quickstart-with-a-HelloWorld-example)
* [Use of camera output](#Use-of-camera-output)
* [Object detection and other tasks](#Object-detection-and-other-tasks)
* [Optimization of performance on mobile](#Optimization-of-performance-on-mobile)

<br>

## Quickstart with a HelloWorld example

[HelloWorld](https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp) is a simple image classification application that demonstrates how to use PyTorch Android API. This application runs TorchScript serialized model on static image.

#### 1. Download from [github](https://github.com/pytorch/android-demo-app)

```
git clone https://github.com/pytorch/android-demo-app.git
cd HelloWorldApp
```

#### 2. Prepare a [TorchScript](https://pytorch.org/docs/stable/jit.html) model

```
python trace_model.py
```

  The serialized and optimized model is saved at "app/src/main/assets/model.pt"

#### 3. Run the model

Open the folder at Android Studio and run the project.

Please, check more detailed [code explanation](https://github.com/pytorch/android-demo-app/tree/master/HelloWorldApp) of this guide.

<details>
<summary>
Troubleshooting
</summary>
<br>
  <details>
  <summary>
  Could not find org.pytorch:pytorch_android:1.8.0-SNAPSHOT.
  </summary>
  <br>
  At <code>build.gradle</code>, remove -SNAPSHOT.
  <pre><code>
  implementation 'org.pytorch:pytorch_android:1.8.0'
  implementation 'org.pytorch:pytorch_android_torchvision:1.8.0'
  </code></pre>
  </details>
  <details>
  <summary>
  error: no suitable method found for bitmapToFloat32Tensor(Bitmap,float[],float[],MemoryFormat)
  </summary>
  <br>
  In the <code>MainActivity.java</code>, remove MemoryFormat.CHANNELS_LAST.
  <pre><code>
  final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
  </code></pre>
  </details>
</details>

<br>

## Use of camera output

You can use [Android CameraX API](https://developer.android.com/training/camerax) to get device camera output as described in [PyTorch Demo Application](https://github.com/pytorch/android-demo-app/tree/master/PyTorchDemoApp). In this demo, models are already prepared inside the app as android asset. Just open in Android Studio and run it.

<br>

## Object detection and other tasks

Simply follow the same steps of [Quickstart](#Quickstart-with-a-HelloWorld-Example).
1. Download a project (e.g., [D2Go](https://github.com/pytorch/android-demo-app/tree/master/D2Go) model, [YOLOv5](https://github.com/pytorch/android-demo-app/tree/master/ObjectDetection)).
2. Prepare a model.
3. Run the project.
<details>
<summary>
Troubleshooting
</summary>
<br>
  <details>
  <summary>
  Failure in building wheel of detectron2 when pip install.
  </summary>
  <br>
  Please check package versions.<br>
  <pre><code>
  cudatoolkit==10.1.243-h6bb024c_0
  pytorch==1.8.0-py3.8_cuda10.1_cudnn7.6.3_0
  torchvision==0.9.0-py38_cu101
  </code></pre>
  I changed cudatoolkit version from 10.2 to 10.1.
  </details>
  <details>
  <summary>
  ERROR: Could not find a version that satisfies the requirement black==21.4b2 (from detectron2)
  </summary>
  <br>
  Install black from other channels. <code>conda install black==21.4b2 -c conda-forge</code>
  </details>
  <details>
  <summary>
  ERROR: Directory '.' is not installable. Neither 'setup.py' nor 'pyproject.toml' found.
  </summary>
  <br/>
  Change the original command <code>cd d2go & python -m pip install .</code> to
  <pre><code>
  cd d2go
  pip install .
  </code></pre>
  </details>
  <details>
  <summary>
  ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
  </summary>
  <br/>
  Reinstall numpy package
  <pre><code>
  pip uninstall numpy
  pip install numpy
  </code></pre>
  </details>
  <details>
  <summary>
  ModuleNotFoundError: No module named 'd2go.tests.data_loader_helper'
  </summary>
  <br/>
  At <code>create_d2go.py</code>, change python code
  <pre><code>
  from d2go.tests.data_loader_helper import LocalImageGenerator, register_toy_dataset
  </code></pre>
  to
  <pre><code>
  from d2go.utils.testing.data_loader_helper import LocalImageGenerator
  from d2go.utils.testing.data_loader_helper import _register_toy_dataset as register_toy_dataset
  </code></pre>
  </details>
  <details>
  <summary>
  RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
  </summary>
  <br/>
  At <code>create_d2go.py</code>, add <code>pytorch_model.cpu()</code> after <code>pytorch_model = model_zoo.get(cfg_name, trained=True)</code>
  </details>
</details>

[PyTorch Android examples](https://github.com/pytorch/android-demo-app) supports more applications such as [image segmentation](https://github.com/pytorch/android-demo-app/tree/master/ImageSegmentation) and its [tutorial](https://pytorch.org/tutorials/beginner/deeplabv3_on_android.html), vision transformer, video classification, speech recognition, neural machine translation, etc.

<br>

## Optimization of performance on mobile

### PyTorch Mobile Performance Recipes

[PyTorch Mobile Performance Recipes](https://pytorch.org/tutorials/recipes/mobile_perf.html) introduces a list of recipes for performance optimizations for using PyTorch on Mobile.

- [Fuse](https://pytorch.org/tutorials/recipes/fuse.html) operators using `torch.quantization.fuse_modules`
- [Quantize](https://pytorch.org/tutorials/recipes/quantization.html) the model
- Use `torch.utils.mobile_optimizer` ([description](https://pytorch.org/docs/stable/mobile_optimizer.html) and [example](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/trace_model.py))
- Use [channels last memory format](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) (NHWC)
- Reusing tensors for forward
- Benchmarking with a naked binary

<details>
<summary>
Issues
</summary>
<br>
  <details>
  <summary>
  Could not build benchmark binary <code>build_android/bin/speed_benchmark_torch</code>.
  </summary>
  <br>
  This is issue is not resolved now. Please refer to these links:<br>
  1. https://pytorch.org/tutorials/recipes/android_native_app_with_custom_op.html<br>
  2. https://github.com/pytorch/pytorch/tree/master/scripts#build_androidsh
  </details>
</details>

### Others

<details>
<summary>
One quick note about the time performance of model on app.
</summary>
<br>
In the <code>MainActivity.java</code>, the following code snippet shows how fast the model runs:
<pre><code>
final long startTime = SystemClock.elapsedRealtime();
IValue[] outputTuple = mModule.forward(IValue.listFrom(inputTensor)).toTuple();
final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
Log.d("D2Go",  "inference time (ms): " + inferenceTime);
</code></pre>
(But, performance behavior can vary in different environments.)
</details>
<details>
<summary>
How to set the number of threads for faster inference.
</summary>
<br>
You can use <code>setNumThreads</code> method at <code><a href="https://github.com/pytorch/pytorch/blob/master/android/pytorch_android/src/main/java/org/pytorch/PyTorchAndroid.java#L40">org.pytorch.PyTorchAndroid</a></code>.<br>
Here is an example in <a href="https://github.com/pytorch/pytorch/blob/master/android/test_app/app/src/main/java/org/pytorch/testapp/MainActivity.java#L132">test app code</a>,
<pre><code>
PyTorchAndroid.setNumThreads(1);
mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), BuildConfig.MODULE_ASSET_NAME);
</code></pre>
or you can see this <a href="https://discuss.pytorch.org/t/android-pytorch-forward-method-running-in-a-separate-thread-slow-down-ui-thread/63516">dicussion</a>.
</details>
<details>
<summary>
(Prototype) Introduction of Lite Interpreter to reduce the runtime binary size.
</summary>
<br>
This is still on progress and a custom pytorch binary from source is required to build libtorch lite for android. Please see this <a href="https://pytorch.org/tutorials/prototype/lite_interpreter.html">tutorial</a>
</details>

<br>

## To do List

The above guide executes the models on the CPU backend and the use of other hardware backends such as GPU, DSP, and NPU is currently in the prototype phase.

- [ ] [GPU support on Android via Vulkan](#PyTorch-with-Vulkan-backend-for-GPU)
- [ ] [DPS and NPU support on Andoir via Google NNAPI](#PyTorch-with-NNAPI)

<br>

## Build PyTorch Android
(Need to update)
- For prerequisites,
  - installation of Java SDK, gradle, Android SDK, and Android NDK ([link1](../../platforms/Android/common.md#Build-and-run-with-command-lines-(for-Linux)), [link2](https://pytorch.org/tutorials/recipes/android_native_app_with_custom_op.html)).
  - [Android build guide](https://pytorch.org/mobile/android/#building-pytorch-android-from-source)

To build LibTorch for android with Vulkan backend for specified <code>ANDOIR_ABI</code>.
```
cd PYTORCH_ROOT
ANDROID_ABI=arm64-v8a USE_VULKAN=1 sh ./scripts/build_android.sh
```
or to prepare pytorch_android aars that you can use directly in your app:
```
cd $PYTORCH_ROOT
USE_VULKAN=1 sh ./scripts/build_pytorch_android.sh
```

<br>

## PyTorch with Vulkan backend for GPU
(NOTICE: This guide is not completed due to errors)

[PyTorch Vulkan Backen User Workflow](https://pytorch.org/tutorials/prototype/vulkan_workflow.html) introduce how to run model inference on GPUs that support the [Vulkan](https://en.wikipedia.org/wiki/Vulkan_(API)) graphics and compute API.

### Prerequisites
- [Conda](../frameworks/conda.md) environment.

### 1. Install Vulkan for Linux
1. Install Vulkan driver for Intel HD Graphics.
```
sudo apt install mesa-vulkan-drivers
```
You could verify there is a .json manifest file located in either /etc/vulkan/icd.d/ or /usr/share/vulkan/icd.d for that driver installed (though other locations are possible).

After that, follow [Getting Started with the Linux Tarball Vulkan SDK](https://vulkan.lunarg.com/doc/sdk/1.2.176.1/linux/getting_started.html).

2. Install the following prerequisite packages:

```
sudo apt install libglm-dev cmake libxcb-dri3-0 libxcb-present0 libpciaccess0 \
libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++ gcc g++-multilib \
libmirclient-dev libwayland-dev libxrandr-dev libxcb-randr0-dev libxcb-ewmh-dev \
git python3 bison libx11-xcb-dev liblz4-dev libzstd-dev
```
Minimum [CMake 3.10.2](https://cmake.org/files/v3.10/cmake-3.10.2-Linux-x86_64.tar.gz) version is required.

3. Download Vulkan [SDK Tarball](https://vulkan.lunarg.com/sdk/home#linux) for Linux

4. Extract the SDK package.
```
cd ~
mkdir vulkan
cd vulkan
tar xf $HOME/Downloads/vulkansdk-linux-x86_64-1.x.yy.z.tar.gz
```

5. Set up the runtime environment
```
source ~/vulkan/1.x.yy.z/setup-env.sh
```
Alternatively you can setup your paths by setting these environment variables yourself:
```
export VULKAN_SDK=~/vulkan/1.x.yy.z/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/etc/vulkan/explicit_layer.d
```

6. Verify the SDK installation
Verify the installation of the Vulkan SDK by running:
```
vkvia
vulkaninfo
vkcube
```

There is also a [Vulkan Guide](https://github.com/KhronosGroup/Vulkan-Guide) github for developers.

### 2. Build PyTorch with Vulkan backend
This is based on [building PyTorch from source](https://github.com/pytorch/pytorch#install-dependencies) on Linux without Vulkan backend.

1. Install dependencies
```
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```

2. Get the PyTorch source
```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

3. Install PyTorch
```
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
cd PYTORCH_ROOT
USE_VULKAN=1 USE_VULKAN_SHADERC_RUNTIME=1 USE_VULKAN_WRAPPER=0 python setup.py install
```

4. After successful build, open another terminal with different path and verify the version of installed PyTorch.
```
>>> import torch
>>> print(torch.__version__)
1.10.0a0+git04986b9
```

### 3. Build Android with Vulkan backend
- For prerequisites,
  - installation of Java SDK, gradle, Android SDK, and Android NDK ([link1](../../platforms/Android/common.md#Build-and-run-with-command-lines-(for-Linux)), [link2](https://pytorch.org/tutorials/recipes/android_native_app_with_custom_op.html)).
  - [Android build guide](https://pytorch.org/mobile/android/#building-pytorch-android-from-source)

To build LibTorch for android with Vulkan backend for specified <code>ANDOIR_ABI</code>.
```
cd PYTORCH_ROOT
ANDROID_ABI=arm64-v8a USE_VULKAN=1 sh ./scripts/build_android.sh
```
or to prepare pytorch_android aars that you can use directly in your app:
```
cd $PYTORCH_ROOT
USE_VULKAN=1 sh ./scripts/build_pytorch_android.sh
```

### 4. Prepare a model
```
import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()
script_model = torch.jit.script(model)
script_model_vulkan = optimize_for_mobile(script_model, backend='vulkan')
torch.jit.save(script_model_vulkan, "mobilenet2-vulkan.pt")
```
The result model can be used only on Vulkan backend as it contains specific to the Vulkan backend operators.

### 5. Using Vulkan backend in code with Android Java API
For Android API to run model on Vulkan backend we have to specify this during model loading:
```
import org.pytorch.Device;
Module module = Module.load("$PATH", Device.VULKAN)
FloatBuffer buffer = Tensor.allocateFloatBuffer(1 * 3 * 224 * 224);
Tensor inputTensor = Tensor.fromBlob(buffer, new int[]{1, 3, 224, 224});
Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();
```

### 6. Building android test app with Vulkan
Check the test application in the PyTorch repository using Vlukan backend and install on the device.
```
# Add prepared model to test application assets:
cp mobilenet2-vulkan.pt $PYTORCH_ROOT/android/test_app/app/src/main/assets/
# build and install
cd $PYTORCH_ROOT
gradle -p android test_app:installMbvulkanLocalBaseDebug
```

<details>
<summary>
Troubleshooting
</summary>
<br>
  <details>
  <summary>
  Failed to install Vulkan driver in doeker container.
  </summary>
  <br>
  You can check this error with <code>vulkaninfo</code> command.
  <pre><code>
  /build/vulkan-UL09PJ/vulkan-1.1.70+dfsg1/demos/vulkaninfo.c:2700: failed with VK_ERROR_INITIALIZATION_FAILED
  </code></pre>
  If Vulkan driver is not installed, please install Vulkan driver at the outside of docker container.
  </details>
  <details>
  <summary>
  USE_VULKAN: Shaderc not found in VULKAN_SDK
  </summary>
  <br>
  At <code>$PYTORCH_ROOT/cmake/VulkanDependencies.cmake</code> Line:162, change
  <pre><code>
  find_library(
    GOOGLE_SHADERC_LIBRARIES
    NAMES shaderc_combined
    PATHS ${GOGGLE_SHADERC_LIBRARY_SERACH_PATH})
  </code></pre>
  to
  <pre><code>
  find_library(
    GOOGLE_SHADERC_LIBRARIES
    NAMES shaderc_combined
    PATHS ${VULKAN_SDK}/x86_64/lib)
  </code></pre>
  </details>
  <details>
  <summary>
  Failed to build PyTorch with Vulkan backend.
  </summary>
  <br>
  This error is not resolved now. The log was:
  <pre><code>
  [3559/5335] Building CXX object caffe2/CMakeFiles/ivalue_test.dir/__/aten/src/ATen/test/ivalue_test.cpp.o
  ../aten/src/ATen/test/ivalue_test.cpp: In member function ‘virtual void c10::IValueTest_getSubValues_Test::TestBody()’:
  ../aten/src/ATen/test/ivalue_test.cpp:593:27: warning: ‘c10::IValue::IValue(std::unordered_map<Key, Value>) [with Key = long int; Value = at::Tensor]’ is deprecated: IValues based on std::unordered_map<K, V> are slow and deprecated. Please use c10::Dict<K, V> instead. [-Wdeprecated-declarations]
    IValue dict(std::move(m));
                            ^
  In file included from ../aten/src/ATen/core/ivalue.h:1138:0,
                  from ../aten/src/ATen/record_function.h:3,
                  from ../aten/src/ATen/Dispatch.h:5,
                  from ../aten/src/ATen/ATen.h:13,
                  from ../aten/src/ATen/test/ivalue_test.cpp:1:
  ../aten/src/ATen/core/ivalue_inl.h:1169:8: note: declared here
  inline IValue::IValue(std::unordered_map<Key, Value> v)
          ^~~~~~
  [3672/5335] Building CXX object caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/Copy.cpp.o
  FAILED: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/Copy.cpp.o 
  /usr/bin/c++ -DCPUINFO_SUPPORTED_PLATFORM=1 -DFMT_HEADER_ONLY=1 -DFXDIV_USE_INLINE_ASSEMBLY=0 -DHAVE_MALLOC_USABLE_SIZE=1 -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DIDEEP_USE_MKL -DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS -DNNP_CONVOLUTION_ONLY=0 -DNNP_INFERENCE_ONLY=0 -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DTH_BLAS_MKL -DUSE_DISTRIBUTED -DUSE_EXTERNAL_MZCRC -DUSE_RPC -DUSE_TENSORPIPE -D_FILE_OFFSET_BITS=64 -Dtorch_cpu_EXPORTS -Iaten/src -I../aten/src -I. -I../ -I../cmake/../third_party/benchmark/include -Icaffe2/contrib/aten -I../third_party/onnx -Ithird_party/onnx -I../third_party/foxi -Ithird_party/foxi -I../torch/csrc/api -I../torch/csrc/api/include -I../caffe2/aten/src/TH -Icaffe2/aten/src/TH -Icaffe2/aten/src -Icaffe2/../aten/src -Icaffe2/../aten/src/ATen -I../torch/csrc -I../third_party/miniz-2.0.8 -I../aten/src/TH -Ivulkan -I../aten/../third_party/catch/single_include -I../aten/src/ATen/.. -Icaffe2/aten/src/ATen -I../caffe2/core/nomnigraph/include -I../third_party/FXdiv/include -I../c10/.. -Ithird_party/ideep/mkl-dnn/include -I../third_party/ideep/mkl-dnn/src/../include -I../third_party/pthreadpool/include -I../third_party/cpuinfo/include -I../third_party/QNNPACK/include -I../aten/src/ATen/native/quantized/cpu/qnnpack/include -I../aten/src/ATen/native/quantized/cpu/qnnpack/src -I../third_party/cpuinfo/deps/clog/include -I../third_party/NNPACK/include -I../third_party/fbgemm/include -I../third_party/fbgemm -I../third_party/fbgemm/third_party/asmjit/src -I../third_party/FP16/include -I../third_party/tensorpipe -Ithird_party/tensorpipe -I../third_party/tensorpipe/third_party/libnop/include -I../third_party/fmt/include -isystem third_party/gloo -isystem ../cmake/../third_party/gloo -isystem ../cmake/../third_party/googletest/googlemock/include -isystem ../cmake/../third_party/googletest/googletest/include -isystem ../third_party/protobuf/src -isystem /home/jinbae/anaconda3/envs/vulkan/include -isystem ../third_party/gemmlowp -isystem ../third_party/neon2sse -isystem ../third_party/XNNPACK/include -isystem /home/jinbae/vulkan/1.2.176.1/x86_64/source/Vulkan-Headers/include -isystem /home/jinbae/vulkan/1.2.176.1/x86_64/include -isystem ../third_party -isystem ../cmake/../third_party/eigen -isystem /home/jinbae/anaconda3/envs/vulkan/include/python3.8 -isystem /home/jinbae/anaconda3/envs/vulkan/lib/python3.8/site-packages/numpy/core/include -isystem ../cmake/../third_party/pybind11/include -isystem ../third_party/ideep/mkl-dnn/include -isystem ../third_party/ideep/include -isystem include -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_VULKAN -DUSE_VULKAN_API -DUSE_VULKAN_SHADERC_RUNTIME -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow -DHAVE_AVX_CPU_DEFINITION -DHAVE_AVX2_CPU_DEFINITION -O3 -DNDEBUG -DNDEBUG -fPIC -DCAFFE2_USE_GLOO -DHAVE_GCC_GET_CPUID -DUSE_AVX -DUSE_AVX2 -DTH_HAVE_THREAD -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-missing-braces -Wno-maybe-uninitialized -fvisibility=hidden -O2 -fopenmp -DCAFFE2_BUILD_MAIN_LIB -pthread -DASMJIT_STATIC -std=gnu++14 -MD -MT caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/Copy.cpp.o -MF caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/Copy.cpp.o.d -o caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/Copy.cpp.o -c ../aten/src/ATen/native/Copy.cpp
  In file included from ../aten/src/ATen/native/vulkan/api/Adapter.h:7:0,
                  from ../aten/src/ATen/native/vulkan/api/api.h:7,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Shader.h:68:54: error: declaration of ‘typedef class at::native::vulkan::api::Handle<VkDescriptorSetLayout_T*, at::native::vulkan::api::destroy_DescriptorSetLayout> at::native::vulkan::api::Shader::Layout::Factory::Handle’ [-fpermissive]
        typedef Handle<VkDescriptorSetLayout, Deleter> Handle;
                                                        ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/api/api.h:5:0,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Common.h:107:7: error: changes meaning of ‘Handle’ from ‘class at::native::vulkan::api::Handle<VkDescriptorSetLayout_T*, at::native::vulkan::api::destroy_DescriptorSetLayout>’ [-fpermissive]
  class Handle final {
        ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/api/Adapter.h:7:0,
                  from ../aten/src/ATen/native/vulkan/api/api.h:7,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Shader.h:159:45: error: declaration of ‘typedef class at::native::vulkan::api::Handle<VkShaderModule_T*, at::native::vulkan::api::destroy_ShaderModule> at::native::vulkan::api::Shader::Factory::Handle’ [-fpermissive]
      typedef Handle<VkShaderModule, Deleter> Handle;
                                              ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/api/api.h:5:0,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Common.h:107:7: error: changes meaning of ‘Handle’ from ‘class at::native::vulkan::api::Handle<VkShaderModule_T*, at::native::vulkan::api::destroy_ShaderModule>’ [-fpermissive]
  class Handle final {
        ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/api/Descriptor.h:6:0,
                  from ../aten/src/ATen/native/vulkan/api/Command.h:6,
                  from ../aten/src/ATen/native/vulkan/api/api.h:8,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Resource.h:171:44: error: declaration of ‘typedef class at::native::vulkan::api::Handle<VkSampler_T*, at::native::vulkan::api::destroy_Sampler> at::native::vulkan::api::Resource::Image::Sampler::Factory::Handle’ [-fpermissive]
          typedef Handle<VkSampler, Deleter> Handle;
                                              ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/api/api.h:5:0,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Common.h:107:7: error: changes meaning of ‘Handle’ from ‘class at::native::vulkan::api::Handle<VkSampler_T*, at::native::vulkan::api::destroy_Sampler>’ [-fpermissive]
  class Handle final {
        ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/api/Command.h:7:0,
                  from ../aten/src/ATen/native/vulkan/api/api.h:8,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Pipeline.h:75:49: error: declaration of ‘typedef class at::native::vulkan::api::Handle<VkPipelineLayout_T*, at::native::vulkan::api::destroy_PipelineLayout> at::native::vulkan::api::Pipeline::Layout::Factory::Handle’ [-fpermissive]
        typedef Handle<VkPipelineLayout, Deleter> Handle;
                                                  ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/api/api.h:5:0,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Common.h:107:7: error: changes meaning of ‘Handle’ from ‘class at::native::vulkan::api::Handle<VkPipelineLayout_T*, at::native::vulkan::api::destroy_PipelineLayout>’ [-fpermissive]
  class Handle final {
        ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/api/Command.h:7:0,
                  from ../aten/src/ATen/native/vulkan/api/api.h:8,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Pipeline.h:134:41: error: declaration of ‘typedef class at::native::vulkan::api::Handle<VkPipeline_T*, at::native::vulkan::api::destroy_Pipeline> at::native::vulkan::api::Pipeline::Factory::Handle’ [-fpermissive]
      typedef Handle<VkPipeline, Deleter> Handle;
                                          ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/api/api.h:5:0,
                  from ../aten/src/ATen/native/vulkan/ops/Common.h:6,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/api/Common.h:107:7: error: changes meaning of ‘Handle’ from ‘class at::native::vulkan::api::Handle<VkPipeline_T*, at::native::vulkan::api::destroy_Pipeline>’ [-fpermissive]
  class Handle final {
        ^~~~~~
  In file included from ../aten/src/ATen/native/vulkan/ops/Common.h:7:0,
                  from ../aten/src/ATen/native/vulkan/ops/Copy.h:5,
                  from ../aten/src/ATen/native/Copy.cpp:8:
  ../aten/src/ATen/native/vulkan/ops/Tensor.h:395:19: error: ‘class at::native::vulkan::ops::vTensor::View::State’ is private within this context
        const View::State::Bundle&);
                    ^~~~~
  ../aten/src/ATen/native/vulkan/ops/Tensor.h:275:11: note: declared private here
      class State final {
            ^~~~~
  [3675/5335] Building CXX object caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/DilatedMaxPool3d.cpp.o
  ninja: build stopped: subcommand failed.
  Traceback (most recent call last):
    File "setup.py", line 818, in <module>
      build_deps()
    File "setup.py", line 315, in build_deps
      build_caffe2(version=version,
    File "/home/jinbae/pytorch/tools/build_pytorch_libs.py", line 58, in build_caffe2
      cmake.build(my_env)
    File "/home/jinbae/pytorch/tools/setup_helpers/cmake.py", line 345, in build
      self.run(build_args, my_env)
    File "/home/jinbae/pytorch/tools/setup_helpers/cmake.py", line 140, in run
      check_call(command, cwd=self.build_dir, env=env)
    File "/home/jinbae/anaconda3/envs/vulkan/lib/python3.8/subprocess.py", line 364, in check_call
      raise CalledProcessError(retcode, cmd)
  subprocess.CalledProcessError: Command '['cmake', '--build', '.', '--target', 'install', '--config', 'Release', '--', '-j', '4']' returned non-zero exit status 1.
  </code></pre>
  </details>
</details>

<br>

## PyTorch with NNAPI
(NOTICE: This guide is not completed due to errors)

This [tutorial](https://pytorch.org/tutorials/prototype/nnapi_mobilenetv2.html) shows how to prepare a computer vision model to use [Android's Neural Networks API (NNAPI)](https://developer.android.com/ndk/guides/neuralnetworks). Please note that PyTorch's NNAPI is currently in the "prototype" phase and only supports [a limited range of operators](https://github.com/pytorch/pytorch/blob/master/torch/backends/_nnapi/serializer.py#L38). You can also check [the performance of NNAPI](https://medium.com/pytorch/pytorch-mobile-now-supports-android-nnapi-e2a2aeb74534).

### 1. Set environment
Install PyTorch and Torchvision with proper versions (the latest trunk).
```
pip install --upgrade --pre --find-links https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html torch==1.8.0.dev20201106+cpu torchvision==0.9.0.dev20201107+cpu
```

### 2. Prepare a model
Prepare a nnapi model using <code>torch.backends._nnapi.prepare.convert_model_to_nnapi</code> function. 
<details>
<summary>
The whole example code from <a href="https://pytorch.org/tutorials/prototype/nnapi_mobilenetv2.html">PyTorch turorial</a> is
</summary>
<br>
<pre><code>
#!/usr/bin/env python
import sys
import os
import torch
import torch.utils.bundled_inputs
import torch.utils.mobile_optimizer
import torch.backends._nnapi.prepare
import torchvision.models.quantization.mobilenet
from pathlib import Path


\# This script supports 3 modes of quantization:
\# - "none": Fully floating-point model.
\# - "core": Quantize the core of the model, but wrap it a
\#    quantizer/dequantizer pair, so the interface uses floating point.
\# - "full": Quantize the model, and use quantized tensors
\#   for input and output.
\#
\# "none" maintains maximum accuracy
\# "core" sacrifices some accuracy for performance,
\# but maintains the same interface.
\# "full" maximized performance (with the same accuracy as "core"),
\# but requires the application to use quantized tensors.
\#
\# There is a fourth option, not supported by this script,
\# where we include the quant/dequant steps as NNAPI operators.
def make_mobilenetv2_nnapi(output_dir_path, quantize_mode):
    quantize_core, quantize_iface = {
        "none": (False, False),
        "core": (True, False),
        "full": (True, True),
    }[quantize_mode]

    model = torchvision.models.quantization.mobilenet.mobilenet_v2(pretrained=True, quantize=quantize_core)
    model.eval()

    # Fuse BatchNorm operators in the floating point model.
    # (Quantized models already have this done.)
    # Remove dropout for this inference-only use case.
    if not quantize_core:
        model.fuse_model()
    assert type(model.classifier[0]) == torch.nn.Dropout
    model.classifier[0] = torch.nn.Identity()

    input_float = torch.zeros(1, 3, 224, 224)
    input_tensor = input_float

    # If we're doing a quantized model, we need to trace only the quantized core.
    # So capture the quantizer and dequantizer, use them to prepare the input,
    # and replace them with identity modules so we can trace without them.
    if quantize_core:
        quantizer = model.quant
        dequantizer = model.dequant
        model.quant = torch.nn.Identity()
        model.dequant = torch.nn.Identity()
        input_tensor = quantizer(input_float)

    # Many NNAPI backends prefer NHWC tensors, so convert our input to channels_last,
    # and set the "nnapi_nhwc" attribute for the converter.
    input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    input_tensor.nnapi_nhwc = True

    # Trace the model.  NNAPI conversion only works with TorchScript models,
    # and traced models are more likely to convert successfully than scripted.
    with torch.no_grad():
        traced = torch.jit.trace(model, input_tensor)
    nnapi_model = torch.backends._nnapi.prepare.convert_model_to_nnapi(traced, input_tensor)

    # If we're not using a quantized interface, wrap a quant/dequant around the core.
    if quantize_core and not quantize_iface:
        nnapi_model = torch.nn.Sequential(quantizer, nnapi_model, dequantizer)
        model.quant = quantizer
        model.dequant = dequantizer
        # Switch back to float input for benchmarking.
        input_tensor = input_float.contiguous(memory_format=torch.channels_last)

    # Optimize the CPU model to make CPU-vs-NNAPI benchmarks fair.
    model = torch.utils.mobile_optimizer.optimize_for_mobile(torch.jit.script(model))

    # Bundle sample inputs with the models for easier benchmarking.
    # This step is optional.
    class BundleWrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod
        def forward(self, arg):
            return self.mod(arg)
    nnapi_model = torch.jit.script(BundleWrapper(nnapi_model))
    torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
        model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])
    torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
        nnapi_model, [(torch.utils.bundled_inputs.bundle_large_tensor(input_tensor),)])

    # Save both models.
    model.save(output_dir_path / ("mobilenetv2-quant_{}-cpu.pt".format(quantize_mode)))
    nnapi_model.save(output_dir_path / ("mobilenetv2-quant_{}-nnapi.pt".format(quantize_mode)))


if __name__ == "__main__":
    for quantize_mode in ["none", "core", "full"]:
        make_mobilenetv2_nnapi(Path(os.environ["HOME"]) / "mobilenetv2-nnapi", quantize_mode)
</code></pre>
</details>

### 3. Test with PyTorchDemoApp
i. Prepare the example android project [PyTorchDemoApp](https://github.com/pytorch/android-demo-app/tree/master/PyTorchDemoApp).

ii. At <code>build.gradle</code>, change the version of pytorch dependencies,
```
  implementation 'org.pytorch:pytorch_android:1.10.0-SNAPSHOT'
  implementation 'org.pytorch:pytorch_android_torchvision:1.10.0-SNAPSHOT'
```
  
At <code>VisionListActivity.java</code>, change <code>intent.putExtra(ImageClassificationActivity.INTENT_MODULE_ASSET_NAME, "model.pt");</code> to <code>intent.putExtra(ImageClassificationActivity.INTENT_MODULE_ASSET_NAME, "mobilenetv2-quant_none-cpu.pt");</code>
  
At <code>ImageClassificationActivity.java</code>, change <code>mModule = Module.load(moduleFileAbsoluteFilePath);</code> to <code>Module module = LiteModuleLoader.load(moduleFileAbsoluteFilePath);</code> (Need to import a proper module)

There was an [issue](https://github.com/pytorch/pytorch/issues/57803) for this change.

iii. Run the project.

<br>

<details>
<summary>
Troubleshooting
</summary>
<br>
  <details>
  <summary>
  ERROR: Could not find a version that satisfies the requirement torch==1.8.0.dev20201106+cpu
  </summary>
  <br>
  You can check the latest version at this <a href="https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html">link</a> and install the package.
  <pre><code>
  pip install --find-links https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html torch==1.10.0.dev20210628+cpu torchvision==0.11.0.dev20210628+cpu
  </code></pre>
  </details>
  <details>
  <summary>
  Exception: Unsupported node kind ('aten::flatten') in node %input : Tensor = aten::flatten(%x, %26, %13) # /opt/conda/envs/nnapi/lib/python3.8/site-packages/torchvision/models/mobilenetv2.py:195:0
  </summary>
  <br>
  This error occurs due to the change of MonileNetV2 code, as written in the <a href="https://github.com/pytorch/pytorch/issues/50533">issue</a>.
  The flatten is not in <a href="https://github.com/pytorch/pytorch/blob/master/torch/backends/_nnapi/serializer.py#L38">supported operations</a> now. You can change the forward pass code from
  <pre><code>
  x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
  x = torch.flatten(x, 1)
  </code></pre>
  to
  <pre><code>
  x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape((x.size(0), -1))
  </code></pre>
  </details>
  <details>
  <summary>
  RuntimeError: Could not run 'quantized::conv2d_relu.new' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend
  </summary>
  <br>
  This error is not resolved now.
  I guess this error occurs when I use the fully quantized torch model, requiring also quantized input. Please check a related <a href="https://discuss.pytorch.org/t/runtimeerror-could-not-run-quantized-conv2d-relu-new-with-arguments-from-the-cpu-backend/106371">thread</a> and the code of function <code><a href="https://github.com/pytorch/pytorch/blob/master/android/pytorch_android_torchvision/src/main/java/org/pytorch/torchvision/TensorImageUtils.java#L261">imageYUV420CenterCropToFloatBuffer</a></code> to make input tensor.
  The full error log was:
  <pre><code>
  2021-06-08 17:07:18.515 11781-11845/org.pytorch.demo E/PyTorchDemo: Error during image analysis
    java.lang.RuntimeException: The following operation failed in the TorchScript interpreter.
    Traceback of TorchScript, serialized code (most recent call last):
      File "code/__torch__/torchvision/models/quantization/mobilenetv2/___torch_mangle_2956.py", line 14, in forward
        else:
          pass
        input = ops.quantized.conv2d_relu(x, CONSTANTS.c0, 0.015018309466540813, 0)
                ~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
        _2 = torch.ne(torch.len(torch.size(input)), 4)
        if _2:
    
    Traceback of TorchScript, original code (most recent call last):
      File "/opt/conda/envs/cpu/lib/python3.8/site-packages/torch/nn/intrinsic/quantized/modules/conv_relu.py", line 85, in forward
                input = F.pad(input, _reversed_padding_repeated_twice,
                              mode=self.padding_mode)
            return torch.ops.quantized.conv2d_relu(
                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                input, self._packed_params, self.scale, self.zero_point)
    RuntimeError: Could not run 'quantized::conv2d_relu.new' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'quantized::conv2d_relu.new' is only available for these backends: [QuantizedCPU, BackendSelect, Named, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, Tracer, Autocast, Batched, VmapMode].
    
    QuantizedCPU: registered at ../aten/src/ATen/native/quantized/cpu/qconv.cpp:873 [kernel]
    BackendSelect: fallthrough registered at ../aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
    Named: registered at ../aten/src/ATen/core/NamedRegistrations.cpp:7 [backend fallback]
    AutogradOther: fallthrough registered at ../aten/src/ATen/core/VariableFallbackKernel.cpp:35 [backend fallback]
    AutogradCPU: fallthrough registered at ../aten/src/ATen/core/VariableFallbackKernel.cpp:39 [backend fallback]
    AutogradCUDA: fallthrough registered at ../aten/src/ATen/core/VariableFallbackKernel.cpp:43 [backend fallback]
    AutogradXLA: fallthrough registered at ../aten/src/ATen/core/VariableFallbackKernel.cpp:47 [backend fallback]
    Tracer: fallthrough registered at ../torch/csrc/jit/frontend/tracer.cpp:999 [backend fallback]
    Autocast: fallthrough registered at ../aten/src/ATen/autocast_mode.cpp:250 [backend fallback]
    Batched: registered at ../aten/src/ATen/BatchingRegistrations.cpp:1016 [backend fallback]
    VmapMode: fallthrough registered at ../aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
    
    
        at org.pytorch.NativePeer.forward(Native Method)
        at org.pytorch.Module.forward(Module.java:49)
        at org.pytorch.demo.vision.ImageClassificationActivity.analyzeImage(ImageClassificationActivity.java:182)
        at org.pytorch.demo.vision.ImageClassificationActivity.analyzeImage(ImageClassificationActivity.java:31)
        at org.pytorch.demo.vision.AbstractCameraXActivity.lambda$setupCameraX$2$AbstractCameraXActivity(AbstractCameraXActivity.java:90)
        at org.pytorch.demo.vision.-$$Lambda$AbstractCameraXActivity$t0OjLr-l_M0-_0_dUqVE4yqEYnE.analyze(Unknown Source:2)
        at androidx.camera.core.ImageAnalysisAbstractAnalyzer.analyzeImage(ImageAnalysisAbstractAnalyzer.java:57)
        at androidx.camera.core.ImageAnalysisNonBlockingAnalyzer$1.run(ImageAnalysisNonBlockingAnalyzer.java:135)
        at android.os.Handler.handleCallback(Handler.java:938)
        at android.os.Handler.dispatchMessage(Handler.java:99)
        at android.os.Looper.loop(Looper.java:223)
        at android.os.HandlerThread.run(HandlerThread.java:67)
  </code></pre>
  </details>
  <details>
  <summary>
  com.facebook.jni.CppException: PytorchStreamReader failed locating file bytecode.pkl: file not found ()
  </summary>
  <br>
  The error log was:
  <pre><code>
  2021-06-30 10:27:39.821 32549-32635/org.pytorch.demo E/PyTorchDemo: Error during image analysis
    com.facebook.jni.CppException: PytorchStreamReader failed locating file bytecode.pkl: file not found ()
    Exception raised from valid at /var/lib/jenkins/workspace/caffe2/serialize/inline_container.cc:157 (most recent call first):
    (no backtrace available)
        at org.pytorch.LiteNativePeer.initHybrid(Native Method)
        at org.pytorch.LiteNativePeer.<init>(LiteNativePeer.java:23)
        at org.pytorch.LiteModuleLoader.load(LiteModuleLoader.java:29)
        at org.pytorch.demo.vision.ImageClassificationActivity.analyzeImage(ImageClassificationActivity.java:168)
        at org.pytorch.demo.vision.ImageClassificationActivity.analyzeImage(ImageClassificationActivity.java:32)
        at org.pytorch.demo.vision.AbstractCameraXActivity.lambda$setupCameraX$2$AbstractCameraXActivity(AbstractCameraXActivity.java:90)
        at org.pytorch.demo.vision.-$$Lambda$AbstractCameraXActivity$t0OjLr-l_M0-_0_dUqVE4yqEYnE.analyze(Unknown Source:2)
        at androidx.camera.core.ImageAnalysisAbstractAnalyzer.analyzeImage(ImageAnalysisAbstractAnalyzer.java:57)
        at androidx.camera.core.ImageAnalysisNonBlockingAnalyzer$1.run(ImageAnalysisNonBlockingAnalyzer.java:135)
        at android.os.Handler.handleCallback(Handler.java:938)
        at android.os.Handler.dispatchMessage(Handler.java:99)
        at android.os.Looper.loop(Looper.java:223)
        at android.os.HandlerThread.run(HandlerThread.java:67)
  </code></pre>
  The default is changed to <a href="https://pytorch.org/tutorials/recipes/mobile_interpreter.html">lite interpreter</a>. You should use <code>script._save_for_lite_interpreter()</code> instead <code>script.save()</code>. You can also check a related <a href="https://github.com/pytorch/android-demo-app/issues/157">github issue</a>.
  </details>
  <details>
  <summary>
  java.lang.UnsatisfiedLinkError: dlopen failed: library "libpytorch_jni.so" not found
  </summary>
  <br>
  The <code>libpytorch_jni.so</code> does not exist in recent <code>pytorch_android</code> <code>SNAPSHOT</code>. (Replaced by <code>libpytorch_jni_lite.so</code>.)
  Please check this <a href="https://github.com/pytorch/pytorch/issues/57803">issue</a>.
  </details>
</details>

=========================================

### 4. Building PyTorch Android from Source (Additional)
If you need to use a local build of PyTorch Android due to any reasons such as local change of code, you may build custom LibTorch binary as guided in this [tutorial](https://pytorch.org/mobile/android/#building-pytorch-android-from-source).

#### 4.1 Dependencies
For the build script, Java SDK, gradle, Android SDK and NDK are required. You need to install them and specify them as environment variables. Below guides show examples.

#### 4.1.1 JAVA_HOME (path to JAVA JDK)
```
sudo apt update
sudo apt install default-jdk
java -version
```
At the end of the file <code>/etc/environment</code>, add the line <code>JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"</code> for a new environment variable. ([link1](https://linuxize.com/post/install-java-on-ubuntu-18-04/), [link2](https://vitux.com/how-to-setup-java_home-path-in-ubuntu/))

#### 4.1.2 GRADLE_HOME (path to gradle)
Prerequisites: Java SDK ([4.1.1](#4.1.1-JAVA_HOME-(path-to-JAVA-JDK)))

i) Download the latest binary-only zip file at [gradle.org](https://gradle.org/releases/) or use wget command and unzip it.
```
wget https://services.gradle.org/distributions/gradle-7.0.2-bin.zip
sudo unzip -d /opt/gradle gradle-7.0.2-bin.zip
```
ii) Make a sh file <code>/etc/profile.d/mygradle.sh</code> and add the following lines to add environment variables:.
```
export GRADLE_HOME=/opt/gradle/gradle-7.0.2/
export PATH=${GRADLE_HOME}/bin:${PATH}
```
iii) Run source command to load this environment variables.
```
sudo chmod u+x /etc/profile.d/mygradle.sh
source /etc/profile.d/mygradle.sh
```
You can check the installation of gradle with <code>gradle -v</code>. Please refer to this [blog](https://www.osetc.com/en/how-to-install-gradle-on-ubuntu-16-04-or-18-04.html) for more information.

#### 4.1.3 ANDROID_HOME (path to Android SDK)
i) Download "command line tools only" zip file at [developer.android.com](https://developer.android.com/studio) and unzip it.
```
sudo mkdir /opt/android
sudo unzip -d /opt/android commandlinetools-linux-7302050_latest.zip
cd /opt/android/cmdline-tools
sudo mkdir tools
mv -i * tools
```
The last command will change the directory structure like this:
```
android
└── cmdline-tools
    └── tools
        ├── NOTICE.txt
        ├── bin
        ├── lib
        └── source.properties
```
ii) Add new environment variables like this:
```
export ANDROID_HOME=/opt/android
export PATH=$ANDROID_HOME/cmdline-tools/tools/bin/:$PATH
export PATH=$ANDROID_HOME/emulator/:$PATH
export PATH=$ANDROID_HOME/platform-tools/:$PATH
```
iii) To verify the setup, run the command <code>sdkmanager --list</code> and you can see the available packages. If there is a package you want to install, just copy the package name and install like <code>sdkmanager --install "package_name"</code>.

There are 4 basic packages you should install like:
```
sudo sdkmanager --install "platform-tools platforms;android-29 build-tools;29.0.2 emulator"
```
This will install all the basic necessary tools you’ll require to start up your android development ([more details](https://proandroiddev.com/how-to-setup-android-sdk-without-android-studio-6d60d0f2812a)).

#### 4.1.4 ANDROID_NDK (path to Android NDK)
i) After you have setup the Android SDK successfully, it is easy to install NDK package through <code>sdkmanager</code>.
```
sudo sdkmanager --install "ndk;22.1.7171670"
```
ii) For environment variable setting, run the following commands. The command <code>which ndk-build</code> shows whether ndk-build is added to environment.
```
export PATH=$PATH:~/android_ndk/android-ndk-r20
export NDK_HOME=~/android_ndk/android-ndk-r20
```
For other installation method using zip file, please refer to this [guide](https://lynxbee.com/how-to-install-android-ndk-on-ubuntu-16-04-18-04/).

#### 4.2 Custom Build
After the installation of all required packages, use the script <code>./scripts/build_pytorch_android.sh</code> to build PyTorch Android. (If sh doesn't work, pleae try source command.)
```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
sh ./scripts/build_pytorch_android.sh
```
After successful build, you should see the result as aar file. And add in the project's app <code>build.gradle</code> file.
```
allprojects {
    repositories {
        flatDir {
            dirs 'libs'
        }
    }
}

dependencies {

    // if using the libraries built from source
    implementation(name:'pytorch_android-release', ext:'aar')
    implementation(name:'pytorch_android_torchvision-release', ext:'aar')

    // if using the nightly built libraries downloaded above, for example the 1.8.0-snapshot on Jan. 21, 2021
    // implementation(name:'pytorch_android-1.8.0-20210121.092759-172', ext:'aar')
    // implementation(name:'pytorch_android_torchvision-1.8.0-20210121.092817-173', ext:'aar')

    ...
    implementation 'com.android.support:appcompat-v7:28.0.0'
    implementation 'com.facebook.fbjni:fbjni-java-only:0.0.3'
}
```
Also we have to add all transitive dependencies of our aars. As <code>pytorch_android</code> depends on <code>com.android.support:appcompat-v7:28.0.0</code> or <code>androidx.appcompat:appcompat:1.2.0</code>, we need to one of them. (In case of using maven dependencies they are added automatically from <code>pom.xml</code>). (Please see [PyTorch Mobile Android](https://pytorch.org/mobile/android/).)

<details>
<summary>
Troubleshooting
</summary>
<br>
  <details>
  <summary>
  Error: Either specify it explicitly with --sdk_root= or move this package into its expected location: <sdk>/cmdline-tools/latest/
  </summary>
  <br>
  Error log from <code>sdkmanager --list</code>:
  <pre><code>
  Error: Could not determine SDK root.
  Error: Either specify it explicitly with --sdk_root= or move this package into its expected location: <sdk>/cmdline-tools/latest/
  </code></pre>
  Please make proper folder structure as mentioned in 4.1.3.
  </details>

  <details>
  <summary>
  CMake Error: The source directory "/home/jinbae/temp/pytorch/build_android_armeabi-v7a/found" does not exist.
  </summary>
  <br>
  Error log was:
  <pre><code>
  CMake Error: The source directory "/home/jinbae/temp/pytorch/build_android_armeabi-v7a/found" does not exist.
  Specify --help for usage, or press the help button on the CMake GUI.
  </code></pre>
  Please install ccache package <code>sudo apt install ccache</code>.
  </details>
  <details>
  <summary>
  ModuleNotFoundError: No module named 'typing_extensions'
  </summary>
  <br>
  Please install <a href="https://github.com/pytorch/pytorch#install-dependencies">dependencies</a> to build PyTorch.
  </details>
  <details>
  <summary>
  torch_android build Configuring incomplete, errors occurred!
  </summary>
  <br>
  Run <code>git submodule update --init --recursive</code>.
  </details>

  <details>
  <summary>
  d
  </summary>
  <br>
  d
  <pre><code>
  </code></pre>
  </details>
</details>
