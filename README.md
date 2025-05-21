<div align=center>

# ⚡️Learning Fused State Representations for Control from Multi-View Observations [ICML 2025]

[![arXiv](https://img.shields.io/badge/arXiv-2502.01316-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/pdf/2502.01316)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-orange?style=flat&logo=huggingface)](https://huggingface.co/datasets/Arya87/MFSC_ICML_2025/tree/main)
[![Checkpoint](https://img.shields.io/badge/Download-Checkpoint-brightgreen?style=flat&logo=google-drive)](https://huggingface.co/datasets/Arya87/MFSC_ICML_2025/tree/main/MFSC_weights)
[![Results](https://img.shields.io/badge/Results-Training%20Logs-orange?style=flat&logo=render)](https://huggingface.co/datasets/Arya87/MFSC_ICML_2025/tree/main/MFSC_results)
[![Project Page](https://img.shields.io/badge/Project-Website-blue?style=flat&logo=github)](https://github.com/zephyr-base/MFSC)

</div>

<p align="center">
  <img src="framework.jpg" alt="Protein Flow Animation" autoplay loop>
</p>

## 🧩 Introduction
This is the code base for our paper on [Learning Fused State Representations for Control from Multi-View Observations]. we propose **M**ulti-view **F**usion **S**tate for **C**ontrol (**MFSC**), firstly incorporating bisimulation metrics learning into MVRL to learn task-relevant representations. Furthermore, we propose a multiview-based mask and latent reconstruction auxiliary task that exploits shared information across views and improves MFSC’s robustness in missing views by introducing a mask token.

## ⚒️ Installation
For installation and setup instructions for each environment, please refer to the corresponding subdirectories under envs/ or their README files. You may also refer to the setup guidelines from [Keypoint3D](https://github.com/buoyancy99/unsup-3d-keypoints) and [DBC](https://github.com/facebookresearch/deep_bisim4control) for additional reference and compatibility.

## 📖 Run Experiments
We evaluate our method on a set of 3D manipulation environments **Meta-World**, a high degree of freedom 3D locomotion environment **PyBullet's Ant**, and a more realistic multi-view highway driving scenario, **CARLA**. 

To train MFSC from scratch on each benchmark, simply execute the corresponding .sh script located in its respective directory:
```
# Meta-world
$ bash run.sh

# Pybullet's Ant
$ bash run.sh

# CARLA
bash run_local_carla096.sh
```

## 🚀 Checkpoints and Original Data
We have made all original training log data, along with intermediate model checkpoints, available in our [Hugging Face repository](https://huggingface.co/datasets/Arya87/MFSC_ICML_2025). We hope this resource is helpful for your experiments and further research.

## 📌 Citation
If you find this work useful for your research, please consider citing it. 
```bibtex
@article{wang2025learning,
  title={Learning Fused State Representations for Control from Multi-View Observations},
  author={Wang, Zeyu and Li, Yao-Hui and Li, Xin and Zang, Hongyu and Laroche, Romain and Islam, Riashat},
  journal={arXiv preprint arXiv:2502.01316},
  year={2025}
}
```

## 👍 Acknowledgments
Thanks to [Keypoint3D](https://github.com/buoyancy99/unsup-3d-keypoints), [DBC](https://github.com/facebookresearch/deep_bisim4control), [SimSR](https://github.com/bit1029public/SimSR) and [MLR](https://github.com/microsoft/Mask-based-Latent-Reconstruction) for their great work and codebase, which served as the foundation for developing MFSC.

## 📧 Contact Us
If you have any question, please feel free to contact us via [zywang0824@bit.edu.cn](mailto:zywang0824@bit.edu.cn).
