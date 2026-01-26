# Night→Day Translation - Projet 9
## CycleGAN pour Embedded AI sur Jetson

> **Projet**: Translation d'images nocturnes vers images diurnes  
> **Dataset**: [huggan/night2day](https://huggingface.co/datasets/huggan/night2day) (Hugging Face)  
> **Modèle**: CycleGAN (unpaired image-to-image translation)  
> **Cible**: NVIDIA Jetson avec démo temps-réel split-screen

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Installation](#installation)
3. [Structure du projet](#structure-du-projet)
4. [Phase 1: Dataset](#phase-1-dataset-preparation)
5. [Phase 2: Training](#phase-2-training-evaluation)
6. [Phase 3: Demo Jetson](#phase-3-demo-jetson)
7. [Résultats attendus](#résultats-attendus)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Vue d'ensemble

Ce projet implémente une **translation night→day** utilisant CycleGAN, permettant de transformer des images nocturnes en images diurnes (ou vice-versa) **sans paires d'images alignées**.

### Caractéristiques
- ✅ **Dataset unpaired** via Hugging Face (pas de téléchargement manuel)
- ✅ **CycleGAN** avec générateurs ResNet-6/9 blocks
- ✅ **Demo temps-réel** sur Jetson avec split-screen (original | processed)
- ✅ **Export ONNX** pour optimisation TensorRT (optionnel)
- ✅ **Mixed precision** (FP16) pour accélération

### Architecture

```
Night Image (A) ──┐
                  │
                  ├──> Generator A→B ──> Fake Day Image (B')
                  │                            │
                  │                            ├──> Discriminator B ──> Real/Fake?
                  │                            │
                  └──> Generator B→A ──> Reconstructed Night (A'')
                                              │
                                         Cycle Loss
```

---

## 🛠 Installation

### Pré-requis système
- Python 3.8+
- CUDA 11.x+ (pour GPU training)
- Git

### 1. Cloner/Créer le repo

```bash
# Créer le dossier du projet
mkdir night2day_project
cd night2day_project

# Copier les fichiers fournis dans ce dossier
```

### 2. Installer les dépendances

#### Sur Windows

```powershell
# Créer environnement virtuel
python -m venv venv
.\venv\Scripts\activate

# Installer PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Installer autres dépendances
pip install -r requirements.txt
```

#### Sur Mac (Apple Silicon)

```bash
# Créer environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer PyTorch (MPS support)
pip install torch torchvision

# Installer autres dépendances
pip install -r requirements.txt
```

#### Sur Jetson (Ubuntu 20.04/22.04)

```bash
# PyTorch pour Jetson: suivre les instructions officielles
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/

# Installer dépendances
pip install -r requirements.txt

# OpenCV est généralement déjà installé sur Jetson
```

### 3. Vérifier l'installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from datasets import load_dataset; print('✓ HuggingFace datasets OK')"
```

---

## 📂 Structure du projet

```
night2day_project/
├── data/
│   └── DATASET_NOTES.txt          # Infos sur le dataset
├── src/
│   ├── configs/
│   │   └── night2day.yaml         # Configuration principale
│   ├── datasets/
│   │   └── unpaired.py            # Dataloader pour unpaired data
│   ├── models/
│   │   └── gan_generator.py       # CycleGAN Generator + Discriminator
│   ├── scripts/
│   │   └── export_onnx.py         # Export ONNX
│   ├── train.py                   # Script d'entraînement
│   ├── eval.py                    # Script d'évaluation
│   └── demo_live_split.py         # Demo temps-réel Jetson
├── requirements.txt
├── README.md                       # Ce fichier
└── report.md                       # Rapport final (à compléter)
```

---

## 📊 Phase 1: Dataset Preparation

### Dataset: huggan/night2day

Le dataset est chargé **automatiquement** via Hugging Face Datasets API.

**Caractéristiques:**
- **Type**: Unpaired (domaines séparés A et B)
- **Domain A**: Images nocturnes
- **Domain B**: Images diurnes
- **Splits**: train, test
- **Format**: RGB images, tailles variables

### Tester le dataloader

```bash
# Test du dataloader (charge depuis Hugging Face)
python src/datasets/unpaired.py
```

**Sortie attendue:**
```
Loading dataset from Hugging Face: huggan/night2day
Loaded 1000+ images per domain for train mode
Batch shape - Domain A (night): torch.Size([4, 3, 256, 256])
✓ Dataloader test successful!
```

### Configuration du dataset

Voir `src/configs/night2day.yaml`:

```yaml
dataset:
  name: "huggan/night2day"
  use_huggingface: true  # Chargement auto depuis HF

training:
  image_size: 256        # Resize à 256x256
  batch_size: 4          # Ajuster selon GPU
```

---

## 🚀 Phase 2: Training & Evaluation

### 2.1 Entraînement

```bash
python src/train.py --config src/configs/night2day.yaml
```

**Paramètres importants** (dans `night2day.yaml`):
- `epochs: 100` - CycleGAN nécessite ~50-100 epochs
- `batch_size: 4` - Ajuster selon GPU (4 pour Jetson, 8-16 pour PC)
- `lambda_cycle: 10.0` - Poids cycle consistency loss
- `lambda_identity: 5.0` - Poids identity loss (optionnel)

**Sortie attendue:**
```
Starting training for 100 epochs
Device: cuda
Batch size: 4
Mixed precision: True

Epoch 0: 100%|████████| 250/250 [02:34<00:00]
  G_total: 12.3456
  G_gan: 0.8234
  G_cycle: 1.1234
✓ Best model saved at epoch 0
```

**Checkpoints sauvegardés:**
- `src/runs/night2day_cyclegan/best.pt` - Meilleur modèle
- `src/runs/night2day_cyclegan/last.pt` - Dernier checkpoint

### 2.2 Évaluation

```bash
python src/eval.py --config src/configs/night2day.yaml \
                    --weights src/runs/night2day_cyclegan/best.pt
```

**Génère:**
- `src/runs/night2day_cyclegan/samples/` - Images de test (input | output)
- `src/runs/night2day_cyclegan/test_metrics.json` - Métadonnées

**Note:** Pour unpaired data, l'évaluation est **qualitative** (pas de PSNR/SSIM).

---

## 🎥 Phase 3: Demo Jetson

### 3.1 Demo temps-réel split-screen

```bash
python src/demo_live_split.py \
    --weights src/runs/night2day_cyclegan/best.pt \
    --size 256
```

**Contrôles:**
- `q` : Quitter
- `s` : Sauvegarder la frame actuelle

**Affichage:**
```
┌─────────────┬─────────────┐
│  Original   │  Night→Day  │
│  (camera)   │  (generated)│
│             │             │
│ FPS: 12.3   │             │
└─────────────┴─────────────┘
```

### 3.2 Optimisation performance

Pour améliorer le FPS:

1. **Réduire la résolution:**
```bash
python src/demo_live_split.py --weights best.pt --size 192  # ou 128
```

2. **Export ONNX** (optionnel):
```bash
python src/scripts/export_onnx.py \
    --weights src/runs/night2day_cyclegan/best.pt \
    --out model.onnx \
    --size 256
```

3. **TensorRT** (avancé, Jetson uniquement):
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

### 3.3 Configuration Jetson

Voir `src/configs/night2day.yaml`:

```yaml
jetson:
  camera_width: 1280
  camera_height: 720
  camera_fps: 30
  inference_size: 256      # Réduire si FPS < 10
  display_fps: true
```

---

## 📈 Résultats attendus

### Métriques

| Métrique | Valeur attendue | Note |
|----------|-----------------|------|
| **Training epochs** | 50-100 | CycleGAN nécessite plus d'epochs |
| **G_total loss** | < 5.0 (final) | Somme GAN + Cycle + Identity |
| **Cycle loss** | < 1.0 (final) | Cohérence cycle A→B→A |
| **FPS Jetson** | 10-20 FPS @ 256px | Dépend du modèle et résolution |
| **Model size** | ~15-20MB | ResNet-6 blocks |
| **Parameters** | ~7-8M | Generator only |

### Checklist qualité

- [ ] Images générées visuellement plausibles (couleurs jour/nuit)
- [ ] Pas d'artefacts majeurs (grille, bruit)
- [ ] Cohérence temporelle en démo live
- [ ] FPS ≥ 10 sur Jetson @ 256px
- [ ] 8-16 samples sauvegardés dans `samples/`

---

## 🐛 Troubleshooting

### Problème: Dataset ne se charge pas

**Erreur:** `ConnectionError` ou timeout

**Solution:**
```bash
# Vérifier connexion Hugging Face
huggingface-cli login

# Ou télécharger manuellement et changer config:
# use_huggingface: false
# domain_a_path: "data/night2day/trainA"
# domain_b_path: "data/night2day/trainB"
```

### Problème: Out of Memory (OOM) GPU

**Solution:**
```yaml
# Dans night2day.yaml, réduire:
batch_size: 2           # au lieu de 4
image_size: 192         # au lieu de 256
base_filters: 32        # au lieu de 64 (generator lite)
```

### Problème: DataLoader lent sur Windows

**Solution:**
```yaml
# Dans night2day.yaml:
num_workers: 0  # au lieu de 4
```

### Problème: Camera CSI ne s'ouvre pas (Jetson)

**Solution:**
```bash
# Vérifier caméra
v4l2-ctl --list-devices

# Fallback USB:
# Le script essaie automatiquement cv2.VideoCapture(0)
```

### Problème: FPS trop faible en demo

**Solutions:**
1. Réduire résolution: `--size 128` ou `--size 192`
2. Export ONNX + TensorRT (voir Phase 3.2)
3. Utiliser modèle lite (base_filters: 32)

---

## 📝 Rapport final

Compléter `report.md` avec:

1. **Dataset & preprocessing**
   - Source, splits, augmentations
2. **Model & loss**
   - Architecture choisie, hyperparamètres
3. **Résultats**
   - Loss curves, samples visuels
4. **Demo Jetson**
   - FPS mesuré, résolution utilisée
5. **Lessons learned**
   - Trade-offs accuracy vs speed
   - Difficultés rencontrées

---

## 📚 Ressources

- **CycleGAN Paper:** https://arxiv.org/abs/1703.10593
- **Dataset:** https://huggingface.co/datasets/huggan/night2day
- **PyTorch Jetson:** https://forums.developer.nvidia.com/t/pytorch-for-jetson/
- **TensorRT:** https://docs.nvidia.com/deeplearning/tensorrt/

---

## 👥 Équipe

- **Préparé par:** [Ton nom]
- **Date:** 2026-01-26
- **Projet:** ESIEA Embedded AI - Night→Day Translation

---

**Bon courage pour le projet ! 🚀**
