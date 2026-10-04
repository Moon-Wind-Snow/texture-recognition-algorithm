```
# Texture‑Recognition‑of‑Vision‑Based‑Tactile‑Sensor
This repository implements material texture recognition for robot end‑effectors using vision‑based tactile sensor images.
The project includes dataset construction, comparative studies of ConvNeXt and DINOv2, a hybrid fusion model TRGFormer, multi‑stage progressive transfer‑learning, and a producer‑consumer multi‑thread asynchronous real‑time inference pipeline.
TRGFormer achieves **90.23 % test accuracy** on our self‑collected 13‑class tactile texture dataset.

## 1. Project Overview
With the rapid advance of embodied intelligence, tactile perception plays a critical role in enabling robots to perform dexterous manipulation. Vision‑based tactile sensors convert physical contact deformation into high‑resolution images, allowing robots to perceive surface textures, roughness, and intrinsic material properties.

This work adopts a vision‑based tactile sensing platform with white‑light illumination and a wrinkle‑free opaque soft elastomer cover, which suppresses reflective noise and artifacts originating from the sensor’s own structure.
A tactile texture dataset is built upon this hardware platform, containing **13 common material categories with more than 6000 samples**. Data is collected under diverse contact conditions including varying pressure, contact angles and sliding motions.

We conduct systematic comparisons between ConvNeXt and DINOv2: ConvNeXt shows strengths in capturing local high‑frequency texture features, while DINOv2 excels at global low‑frequency structural modeling.
A hybrid model **TRGFormer (Texture‑enhanced Register‑guided Fusion Former)** is proposed. It combines a ConvNeXt local texture branch with a DINOv2 global backbone, utilizing register tokens and gated fusion mechanisms to aggregate both local fine‑grained details and global structural information.

To resolve performance bottlenecks in offline‑to‑online deployment, a **producer‑consumer multi‑thread asynchronous pipeline** is implemented for real‑time tactile material recognition, raising system throughput from 10‑12 FPS up to 25‑30 FPS.

## 2. Main Features
- ✅ Tactile image preprocessing pipeline: center‑crop, resize, grayscale conversion, CLAHE contrast enhancement, data augmentation.
- ✅ Dataset partition strategy: split by physical sample / video sequence to avoid data leakage (train/val/test ≈ 8:1:1).
- ✅ Multi‑stage progressive transfer‑learning training framework for ConvNeXt / DINOv2 / TRGFormer.
- ✅ Visualization toolkit: Grad‑CAM activation heatmap, FFT frequency‑domain analysis, PCA / t‑SNE feature embedding, confusion matrix, training curves.
- ✅ Frequency‑perturbation robustness evaluation (Gaussian blur, low‑pass / high‑pass filter).
- ✅ TRGFormer hybrid fusion model: local texture injection, register tokens, adaptive gated fusion.
- ✅ Producer‑consumer multi‑thread asynchronous real‑time inference system with non‑blocking frame‑buffer queue.
- ✅ Quantitative evaluation: Accuracy, Precision, Recall, F1‑score, confusion‑matrix, learning curves.

## 3. Overall Workflow
1. Tactile data acquisition from vision‑based tactile sensor
2. Image pre‑processing & offline augmentation
3. Dataset splitting (split by sample instance, avoid frame‑level random split leakage)
4. Model training with multi‑stage progressive transfer‑learning
5. Offline evaluation, frequency‑domain & visualization analysis
6. Export checkpoint & deploy to multi‑thread real‑time inference pipeline
7. Real‑world online material‑recognition validation on tactile stream

## 4. Technical Highlights
### 4.1 Vision‑based Tactile Data & Preprocessing
Raw frames captured by the tactile sensor contain redundant borders and uneven illumination. We perform center‑cropping to retain valid contact area, resize to unified input resolution, optionally convert to grayscale, and apply CLAHE for local contrast enhancement.
Training‑time augmentations include random crop, rotation, horizontal flip, random erasing to simulate real‑world contact variation.

> Dataset statistics: 13 material classes (silk, synthetic‑leather, chemical‑fiber, medical‑gauze, nylon, sherpa‑fleece, wooden‑stick, toothbrush, jeans, woven‑bag, tennis‑ball, sports‑shirt, knit‑sweater), >6000 frames.

### 4.2 Multi‑Stage Progressive Transfer‑Learning
Three‑phase training workflow mitigates over‑fitting when training on relatively small tactile datasets:
1. **Stage‑1 (Freeze backbone)**: Only train classification head, lr=1e‑3, epoch=10. Adapt pre‑trained feature space to tactile categories.
2. **Stage‑2 (Partial unfreeze)**: Unfreeze partial backbone layers, lr=5e‑5, epoch=20. Adapt high‑level features to tactile texture.
3. **Stage‑3 (Full fine‑tune)**: Unfreeze all parameters with small learning‑rate, lr=1e‑5, epoch=30. Final joint optimization.

> For TRGFormer hybrid model: stage‑1 only optimizes ConvNeXt branch & fusion modules; stage‑2 unlock partial DINOv2 blocks; stage‑3 full‑model fine‑tune.

### 4.3 Model Comparison & TRGFormer Hybrid Architecture
- **ConvNeXt**: Inductive bias towards **local high‑frequency texture**, fast convergence speed, sensitive to micro‑texture details; limited capacity for global long‑range modeling.
- **DINOv2 (ViT‑based self‑supervised)**: Strong **global low‑frequency structural modeling capability**, good overall category equilibrium; less sensitive to fine‑grained high‑frequency local cues.

**TRGFormer**:
1. ConvNeXt branch extracts local high‑frequency texture features.
2. Feature projection & spatial alignment, inject texture tokens into intermediate layer of DINOv2.
3. Introduce learnable register‑tokens to expand feature aggregation space.
4. Gated‑fusion MLP adaptively adjusts contribution of local texture and global representations.

Quantitative results on test‑set:
| Model | Accuracy | F1‑score |
|---|---|---|
| ConvNeXt | 85.76 % | 86.28 % |
| DINOv2 | 89.21 % | 89.27 % |
| **TRGFormer(Ours)** | **90.23 %** | **90.11 %** |

### 4.4 Producer‑Consumer Real‑Time Inference Pipeline
Serial execution suffers from speed mismatch between camera acquisition (~30 FPS) and model inference latency (~50 ms per frame).
We decouple acquisition / inference / rendering by multi‑thread design:
- Producer thread: camera capture, write frames into bounded non‑blocking queue (`maxsize=8`; drop oldest frame when full to avoid blocking).
- Multiple consumer threads: GPU inference workers consume frame queue.
- Main thread: fetch inference result‑queue and render output.

Performance gain: from **10‑12 FPS (single‑thread serial) → 25‑30 FPS multi‑consumer pipeline**.

## 5. Environment Requirements
Recommended environment:
- Python >= 3.10
- PyTorch 2.7.0 (CUDA 12.8)
- torchvision 0.22.0 (CUDA 12.8)
- numpy, opencv‑python, scikit‑image, scikit‑learn
- matplotlib, seaborn, scipy, tqdm
- timm, transformers

Install dependencies:
```bash
# create & activate virtual environment
python -m venv .venv
source .venv/bin/activate

# install core python packages
pip install -r requirements.txt
```

Core pinned versions in [requirements.txt](requirements.txt):

```
numpy==2.2.6
opencv-python==5.0.0.93
scikit-image
scikit-learn==1.7.2
matplotlib==3.10.9
seaborn==0.13.2
pillow==12.3.0
scipy==1.15.3
tqdm==4.70.1
timm==1.0.29
transformers==5.17.0
```

> 
> ⚠️ Important note:
> `torch` and `torchvision` are **NOT included in requirements.txt**.
> CUDA‑specific wheel tags like `+cu128` cannot be resolved by plain pip.
> Please install PyTorch from the [official PyTorch website](https://pytorch.org/) matching your CUDA / CPU environment.
> An optional conda environment backup file `environment.yml` is also provided for local reference only.

## 6. Repository Structure

```
texture‑recognition‑visuotactile/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── environment.yml               # Conda environment backup (local reference)
├── run.py                         # Main program entry
├── ConvNeXt_DINOv2/               # TRGFormer hybrid fusion model code
│   ├── best_fusion_model.pth      # Fusion model checkpoint
│   ├── eval.py                    # Evaluation script for fusion model
│   ├── fusion_confusion_matrix.png
│   ├── torch_hub/
│   └── train.py                   # Train fusion TRGFormer model
├── ResNet18_convetional/          # ResNet18 baseline experiment
│   ├── confusion_matrix.png
│   ├── eval.py
│   └── train.py
├── all_textured_groups/           # Dataset preprocessing & raw data folder
│   ├── best_model.pth
│   ├── dataset/
│   ├── extract_full_cont.py      # Frame extraction script
│   ├── extract_mid_cont.py
│   ├── test_original_video/
│   └── train_and_val_original_video/
├── including_non_textured_groups/ # Extended dataset group
└── outputs/                       # Generated figures, logs (runtime‑created)
```

> 
> Note:
> 
> 
> - Model checkpoint `.pth` weight files, raw video / dataset files are excluded via `.gitignore`, **do not push large weights and raw data to GitHub**.
> - `outputs/` directory will be automatically generated at runtime.

## 7. Typical Usage

### 7.1 Dataset frame extraction

```
# Extract valid tactile frames from raw video sequences
python all_textured_groups/extract_full_cont.py
python all_textured_groups/extract_mid_cont.py
```

### 7.2 Train TRGFormer fusion model (ConvNeXt + DINOv2)

```
cd ConvNeXt_DINOv2
python train.py
```

### 7.3 Evaluate fusion model

```
cd ConvNeXt_DINOv2
python eval.py
```

### 7.4 Run main program / real‑time pipeline

```
python run.py
```

> 
> For ResNet18 baseline experiment:

```
cd ResNet18_convetional
python train.py
python eval.py
```

## 8. Training & Deployment Notes

- Dataset split must be operated on **sample‑instance / video‑sequence level**, not simple frame‑wise random split, to prevent data‑leakage and over‑optimistic metrics.
- Multi‑stage progressive transfer‑learning is critical for avoiding destroying pre‑trained weights on small tactile dataset.
- When deploying real‑time pipeline: tune queue `maxsize` to balance latency and anti‑jitter; drop oldest frame when queue is full for real‑time scenario.
- Robustness test: Gaussian blur / low‑pass / high‑pass filtering can help analyze model frequency‑domain preference.

## 9. Expected Outcomes

- Off‑line high‑accuracy material texture classification on vision‑based tactile images.
- Mechanism analysis of ConvNeXt / DINOv2 on tactile task: local high‑frequency vs global low‑frequency representation complementarity.
- TRGFormer hybrid model combining CNN local texture and Transformer global modeling.
- A complete real‑time multi‑thread inference solution for robot vision‑based tactile perception.
- Baseline and code reference for further visuo‑tactile perception research.

## 10. Future Directions

- Light‑weight variant of TRGFormer for edge‑device deployment.
- Introduce temporal sequence modeling for continuous tactile sliding frames.
- Expand dataset with more material categories and diverse sensor conditions.
- Combine tactile perception with robot closed‑loop grasping control.
- Explore sim‑to‑real transfer for vision‑based tactile data generation.

## 11. License

This project is released under the [MIT License](LICENSE).
You are free to use, modify and redistribute for research and educational purposes.
See `LICENSE` for full license terms.

## 12. Contact

For questions, discussions or collaboration, please open an issue in this repository.
