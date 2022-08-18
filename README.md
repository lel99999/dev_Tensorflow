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

*** Ran into errors *** <br/>
Solution was to install cuda-10.0, cuda-10.1, cuda-10.2 with --override switch for compiler check.
Otherwise, script will have errors with libraries.

#### Updated - 08/2022 with Tesla P4, CUDA 11.7.1_515.65.01, NVIDIA-Linux-x86_64-515.65.01

#### Cannot find Kernel Headers Error
Solution: <br/>
```
$sudo yum install "kernel-devel-uname-r == $(uname -r)"
```
##### Note: nvidia-docker v2 uses --runtime=nvidia instead of --gpus all. nvidia-docker v1 uses the nvidia-docker alias, rather than the --runtime=nvidia or --gpus all command line flags.

#### Instructions and Commands
- `$sudo docker run --rm --runtime=nvidia -ti nvidia/cuda:11.0.3-base-ubuntu20.04` <br/>
  ![nvidia docker](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-docker-01.png) <br/>

- `$sudo docker run -it --runtime=nvidia --shm-size=1g -e --rm nvcr.io/nvidia/pytorch:18.05-py3` <br/>
  ![nvidia docker pytorch](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-docker-pytorch-01.png) <br/>
  
  Run MNIST traning example with PyTorch:  <br/>
  ```
  root@19a35891107c:/workspace# nvidia-smi
  root@19a35891107c:/workspace# cd examples/mnist
  root@19a35891107c:/workspace/examples/mnist# python main.py
  ```
  ![nvidia pytorch container - MNIST Training Example](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-docker-pytorch-02.png) <br/>
  ![nvidia pytorch container - MNIST Training complete](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-docker-pytorch-03.png) <br/>
  
- Run Tensorflow Images
  ```
  $docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu \
   python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
  
  *** It can take a while to set up the GPU-enabled image. If repeatedly running GPU-based scripts, you can use docker exec to reuse a container.
  
  $docker run --gpus all -it tensorflow/tensorflow:latest-gpu bash
  
  ```
  ![nvidia tensorflow](https://github.com/lel99999/dev_NVIDIA-Docker/blob/master/nvidia-docker-tensorflow-01.png) <br/>
