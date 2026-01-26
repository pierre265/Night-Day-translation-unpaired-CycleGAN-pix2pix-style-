# 🚀 Guide d'Installation Rapide - Night→Day Project

Guide pour démarrer rapidement le projet Night→Day Translation.

---

## ⚡ Installation en 5 minutes

### Windows

```powershell
# 1. Créer environnement virtuel
python -m venv venv
.\venv\Scripts\activate

# 2. Installer PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Installer dépendances
pip install -r requirements.txt

# 4. Vérifier installation
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

### Mac (Apple Silicon)

```bash
# 1. Créer environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 2. Installer PyTorch (MPS support)
pip install torch torchvision

# 3. Installer dépendances
pip install -r requirements.txt

# 4. Vérifier installation
python -c "import torch; print(f'PyTorch {torch.__version__} - MPS: {torch.backends.mps.is_available()}')"
```

**Note Mac:** Dans `src/configs/night2day.yaml`, changer:
```yaml
hardware:
  device: "mps"  # au lieu de "cuda"
```

### Jetson (pour la Phase 3)

```bash
# 1. PyTorch pour Jetson - suivre:
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/

# 2. Installer dépendances
pip3 install -r requirements.txt

# 3. Vérifier
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

---

## 🧪 Tests rapides

### 1. Tester le dataloader

```bash
python src/datasets/unpaired.py
```

**Sortie attendue:**
```
Loading dataset from Hugging Face: huggan/night2day
Loaded 1000+ images per domain for train mode
✅ Dataloader test successful!
```

### 2. Tester le modèle

```bash
python src/models/gan_generator.py
```

**Sortie attendue:**
```
Generator parameters: 7,123,456 (~7.12M)
Discriminator parameters: 2,765,824 (~2.77M)
✅ Model test successful!
```

---

## 📋 Checklist avant de commencer

- [ ] Python 3.8+ installé
- [ ] GPU NVIDIA avec CUDA (recommandé) OU Apple Silicon avec MPS
- [ ] 8GB+ RAM
- [ ] 10GB+ espace disque libre
- [ ] Connexion internet (pour télécharger dataset)

---

## 🏃 Lancer l'entraînement

```bash
# Training (Phase 2)
python src/train.py --config src/configs/night2day.yaml
```

**Premiers epochs attendus:**
```
Starting training for 100 epochs
Device: cuda
Batch size: 4

Epoch 0: 100%|████████| 250/250 [02:34<00:00]
  G_total: 12.3456
  G_gan: 0.8234
  G_cycle: 1.1234
```

**Checkpoints sauvegardés dans:** `src/runs/night2day_cyclegan/`

---

## 🔧 Problèmes courants

### ❌ Erreur: "CUDA out of memory"

**Solution:**
```yaml
# Dans src/configs/night2day.yaml
training:
  batch_size: 2  # réduire de 4 → 2
  image_size: 192  # réduire de 256 → 192
```

### ❌ Erreur: "DataLoader worker crashed" (Windows)

**Solution:**
```yaml
# Dans src/configs/night2day.yaml
hardware:
  num_workers: 0  # mettre à 0
```

### ❌ Dataset ne se télécharge pas

**Solution:**
```bash
# Login Hugging Face
pip install huggingface-hub
huggingface-cli login

# Ou télécharger manuellement depuis:
# https://huggingface.co/datasets/huggan/night2day
```

---

## 📚 Prochaines étapes

1. ✅ **Installation** - Vous êtes ici
2. 📊 **Phase 1** - Dataset (voir `data/DATASET_NOTES.txt`)
3. 🚀 **Phase 2** - Training (voir `README.md` section Phase 2)
4. 🎥 **Phase 3** - Demo Jetson (voir `README.md` section Phase 3)
5. 📝 **Rapport** - Compléter `report.md`

---

## 💬 Besoin d'aide?

1. Consulter le `README.md` complet
2. Vérifier la section Troubleshooting
3. Lire `data/DATASET_NOTES.txt` pour infos dataset
4. Consulter les commentaires dans le code

---

**Bon courage! 🎯**
