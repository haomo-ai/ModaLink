# [IROS 2024] ModaLink

This repository contains the implementation of our IROS 2024 paper:

**ModaLink: Unifying Modalities for Efficient Image-to-PointCloud Place Recognition**

[Weidong Xie](https://sites.google.com/view/dong-hao/), [Lun Luo](https://zjuluolun.github.io/), Nanfei Ye, Yi Ren, Shaoyi Du, Minhang Wang, Jintao Xu, Rui Ai, Weihao Gu and [Xieyuanli Chen](https://github.com/Chen-Xieyuanli)

[Link](https://arxiv.org/abs/2403.18762) to the arXiv version of the paper is available.

The main contributions of this work are:

*  We propose a lightweight cross-modal place recognition method called ModaLink based on FoV transformation. 
*  We introduce a Non-Negative Matrix Factorization-based module to extract extra potential semantic features to improve the distinctiveness of descriptors.
*  Extensive experimental results on the KITTI and a self-collected dataset show that our proposed method can achieve state-of-the-art performance while running in real-time of about 30Hz.

![main](https://github.com/haomo-ai/ModaLink/assets/47657625/28ab99f7-2eaa-4d96-9a7a-918719a69b8d)

## Citation
If you use our implementation in your academic work, please cite the corresponding [paper](https://arxiv.org/abs/2403.18762):
```
@inproceedings{xie2024modalink,
	author   = {Weidong Xie and Lun Luo and Nanfei Ye and Yi Ren and Shaoyi Du and Minhang Wang and Jintao Xu and Rui Ai and Weihao Gu and Xieyuanli Chen},
	title    = {{ModaLink: Unifying Modalities for Efficient Image-to-PointCloud Place Recognition}},
	booktitle  = {In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	year     = {2024},
}
```

## Dependencies

We use pytorch-gpu for neural networks. An Nvidia GPU is needed for faster retrieval.

To use a GPU, first, you need to install the Nvidia driver and CUDA.

- CUDA Installation guide: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

