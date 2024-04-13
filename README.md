# ModaLink

**ModaLink: Unifying Modalities for Efficient Image-to-PointCloud Place Recognition**

*  We propose a lightweight cross-modal place recognition method called ModaLink based on FoV transformation. 
*  We introduce a Non-Negative Matrix Factorization-based module to extract extra potential semantic features to improve the distinctiveness of descriptors.
*  Extensive experimental results on the KITTI and a self-collected dataset show that our proposed method can achieve state-of-the-art performance while running in real-time of about 30Hz.

![main](https://github.com/haomo-ai/ModaLink/assets/47657625/28ab99f7-2eaa-4d96-9a7a-918719a69b8d)


## Table of Contents

## Dependencies

We use pytorch-gpu for neural networks. An Nvidia GPU is needed for faster retrieval.

To use a GPU, first, you need to install the Nvidia driver and CUDA.

- CUDA Installation guide: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

