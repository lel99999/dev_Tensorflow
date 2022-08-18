# dev_Tensorflow
Tensorflow Development and Notes


#### Step-by-Step Configuration Guide for NVIDIA GPU
1) Install GPU
2) Install GPU Driver (440.64) and CUDA Toolkit (10.2)
- Verify Hardware and Software
```
$nvidia-smi
```
![nvidia-smi cmd](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-smi-02.png) <br/>

4) Install nvidia-docker2 and reload the Docker daemon configuration
```
$sudo apt-get install -y nvidia-docker2
$sudo pkill -SIGHUP dockerd
```
5) Test nvidia-smi with the latest official CUDA image that works with your GPU (10.2)
```
$docker run --runtime=nvidia --rm nvidia/cuda:10.2-base nvidia-smi
```
![nvidia-smit test](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-smi_testimage-01.png) <br/>

#### Get NVIDIA Toolkit and Cudnn8 Working on Ubuntu 19.10, Install and Run Tensorflow 2.2
Ran following python code (as tesorflow2_test.py):
```
import tensorflow as tf
print(tf.__version__)

tf.test.is_gpu_available(
    cuda_only = True,
    min_cuda_compute_capability=None
)

print(tf.test.is_gpu_available)

```

##### Issues #####
- Illegal instruction 4
  - [https://github.com/TomHeaven/tensorflow-osx-build/issues/8](https://github.com/TomHeaven/tensorflow-osx-build/issues/8) <br/>

```
$SYSTEM_VERSION_COMPAT=0 pip3 install tensorflow-macos tensorflow-metal
```

#### v1 Notes


#### v2 Notes
