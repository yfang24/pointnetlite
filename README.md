# PointNetLite: Lightweight 3D Object Classification

This repository contains a modular framework for training and evaluating lightweight 3D point cloud classification models. It supports partial and multi-view representation learning on datasets such as ModelNet40 and ScanObjectNN.

## 📁 Project Structure

```
/code
│
├── datasets/           # Dataset wrappers (e.g., modelnet.py, scanobjectnn.py)
├── models/             # Model architectures (e.g., pointnet_encoder.py, cls_head/)
├── utils/              # Utility modules (mesh_utils.py, pcd_utils.py, train_utils.py)
├── config/             # JSON configuration files
├── train.py            # Training script
├── test.py             # Evaluation script
└── README.md           # This file
```

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pointnetlite.git
cd pointnetlite
```

### 2. Create and activate conda environment
```bash
conda create -n pointnetlite-env python=3.9
conda activate pointnetlite-env
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> For PyTorch3D compatibility, please refer to the official installation guide: https://github.com/facebookresearch/pytorch3d

## 🚀 Usage

### Training
```bash
PYTHONPATH=. python train.py --config config/train_config.json
```

### Evaluation
```bash
PYTHONPATH=. python test.py --config config/test_config.json
```

## 📊 Example Results

| Model         | Dataset      | Accuracy (%) |
|---------------|--------------|--------------|
| PointNetLite  | ModelNet40   | 92.4         |
| PointNetLite  | ScanObjectNN | 85.3         |

## 🔍 Features

- Multi-view and partial point cloud learning
- Frozen and learnable encoder fusion
- Rotation and scale augmentation strategies
- Training logs: accuracy, loss, FLOPs, memory, time
- Modular wrappers for datasets and models

## 🧪 Datasets Supported

- ModelNet40 / ModelNet10
- ScanObjectNN (with 11-class subset)
- Sim2Sim and Real2Real benchmarks

## 📝 Citation

If you use this codebase in your research, please cite:

```bibtex
@inproceedings{yourpaper2025,
  title={PointNetLite: Efficient Partial-View 3D Object Recognition},
  author={Your Name and Ping Wang},
  booktitle={SPIE Photonics for Quantum},
  year={2025}
}
```

## 👤 Authors

- Your Name (PhD Student, Stevens Institute of Technology)
- Advisor: Prof. Ping Wang

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
