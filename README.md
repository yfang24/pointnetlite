# PointNetLite: Lightweight 3D Object Classification

This repository contains a modular framework for training and evaluating lightweight 3D point cloud classification models on datasets such as ModelNet40 and ScanObjectNN.

## 📁 Project Structure

```
your_project_root/
├── pointnetlite/
│   ├── configs/
│   ├── datasets/
│   ├── experiments/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   ├── test.py
│   └── __init__.py
└── data/
    ├── modelnet40/
    └── scanobjectnn/
```

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yfang24/pointnetlite.git
cd pointnetlite
```

### 2. Create and activate the environment
✅ We use **Python 3.9.18** and **CUDA 12.4**.

**Using conda:**
```bash
conda create -n pointnetlite-env python=3.9
conda activate pointnetlite-env
```

**Or using venv**
```bash
python3.9 -m venv ~/pointnetlite-env
source ~/pointnetlite-env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Datasets

Download and organize datasets under your_project_root/data/:

- **ModelNet40**: https://modelnet.cs.princeton.edu/  
  - For the aligned version, refer to https://github.com/lmb-freiburg/orion

- **ScanObjectNN**: https://github.com/hkust-vgd/scanobjectnn


## 🚀 Usage

### Set Environment Variables
```bash
export PYTHONPATH=.  # Set Python path
```

If using **multi-node and multi-GPU training** with **DistributedDataParallel (DDP)**, set the master address and port as follows:

#### 🔄 For SLURM Users
```bash
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355  # Use an available, free port
```

#### 💻 For Non-SLURM Users
```bash
export MASTER_ADDR=localhost    # Or IP address of the master node
export MASTER_PORT=12355        # Must match across all processes
```

### Training
```bash
python train.py --cfg config_name
```

Results will be stored in your_project_root/pointnetlite/experiments/exp_name/, including:

- `checkpoint_best.pth`
- `checkpoint_last.pth`
- `config.yaml`
- `log.txt`

To resume training:
```bash
python train.py --exp exp_name
```

### Evaluation
```bash
python test.py -exp exp_name -ckpt checkpoint_type -cm
```
- `--ckpt`: choose from `best` or `last`  
- `--cm`: include this flag to show the confusion matrix

## 📊 Example Results

| Model         | Scenario              | Dataset Description                                        | Accuracy (%) |
|---------------|-----------------------|------------------------------------------------------------|--------------|
| PointNetLite  | Sim2Sim               | ModelNet40                                                 | 87.12        |
| PointNetLite  | Real2Real             | ScanObjectNN (main_split_nobg)                             | 79.00        |
| PointNetLite  | Sim2Real              | Common 11 classes between ModelNet40 and ScanObjectNN      | 65.89        |

## 🔍 Features

- Lightweight and fast **PointNetLite** architecture
- Supports **multi-node, multi-GPU training** via PyTorch **DistributedDataParallel (DDP)**
- Supports **single- and multi-view** point cloud learning
- Effective **Sim2Real adaptation** strategies
- Augmentations: **rendering**, **rotation** and **realistic scaling**
- Logs: accuracy, loss, time, memory, FLOPs

## 📝 Citation

If you use this codebase in your research, please cite:

```bibtex

```

This project builds on several foundational 3D deep learning models. If you use those components, please also consider citing:

```bibtex
@inproceedings{qi2017pointnet,
  title     = {PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author    = {Qi, Charles R. and Su, Hao and Mo, Kaichun and Guibas, Leonidas J.},
  booktitle = {CVPR},
  year      = {2017}
}

@inproceedings{qi2017pointnetplusplus,
  title     = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author    = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
  booktitle = {NeurIPS},
  year      = {2017}
}

@inproceedings{wang2019dgcnn,
  title     = {Dynamic Graph CNN for Learning on Point Clouds},
  author    = {Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
  booktitle = {ACM Transactions on Graphics (TOG)},
  year      = {2019}
}

@article{ma2022pointmae,
  title     = {Point-MAE: Masked Autoencoders for Point Cloud Self-supervised Learning},
  author    = {Ma, Xu and Guo, Yujing and Huang, Xingyu and Bi, Xiang and Chen, Haoyu and Han, Junwei and Tai, Yu-Wing and Tang, Chi-Keung},
  journal   = {arXiv preprint arXiv:2203.06604},
  year      = {2022}
}
```

## 🔗 Acknowledgments

This codebase includes adapted implementations of the following models:

- [PointNet and PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)

We thank the authors of these repositories for open-sourcing their work. Our implementations are adapted and restructured to fit the modular training and evaluation pipeline used in this project.
