# PointNetLite: Lightweight 3D Object Classification

This repository contains a modular framework for training and evaluating lightweight 3D point cloud classification models. It supports partial and multi-view representation learning on datasets such as ModelNet40 and ScanObjectNN.

## Notes


## 📁 Project Structure

```
project/
│
├── code/               # ← contains Python code
│   ├── datasets/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   ├── test.py
│   └── __init__.py
│
├── data/               # ← datasets, cached files, etc.
└── README.md
```

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YFang24/pointnetlite.git
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

### 4. Prepare datasets
download scanobjectnn from .. and store under /data/scanobjectnn
modelnet

## 🚀 Usage

cd code
export PYTHONPATH=.

### Training
```bash
cd code
PYTHONPATH=. python train.py --config config/train_config.json
```

### Evaluation
```bash
cd code
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
