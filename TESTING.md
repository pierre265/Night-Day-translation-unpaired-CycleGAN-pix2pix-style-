# 🧪 TESTING & DEBUGGING GUIDE

**Guide de test et débogage pour le projet Night→Day**  
**À lire AVANT de distribuer aux collègues**

---

## ⚠️ AVERTISSEMENT IMPORTANT

Ce projet est une **base solide et professionnelle** mais n'a **PAS été exécuté de bout en bout** sur une machine réelle. Certains ajustements mineurs seront probablement nécessaires.

**Ce guide vous aidera à :**
1. ✅ Identifier rapidement ce qui doit être adapté
2. ✅ Débugger efficacement les problèmes
3. ✅ Adapter le code selon vos contraintes

---

## 📋 CHECKLIST DE TESTS OBLIGATOIRES

### ✅ PHASE 0: Vérification environnement (2 min)

```bash
python check_environment.py
```

**Résultat attendu :**
- ✅ PyTorch installé
- ✅ CUDA ou MPS disponible (ou accepter CPU lent)
- ✅ Tous les packages présents

**Si échec :**
→ Réinstaller les dépendances : `pip install -r requirements.txt`

---

### ✅ TEST 1: Dataset (CRITIQUE - 5 min)

**Objectif :** Vérifier que le dataset se charge correctement

```bash
python src/datasets/unpaired.py
```

#### Cas 1 : ✅ Succès
```
Loading dataset from Hugging Face: huggan/night2day
Loaded 1234 images per domain for train mode
Batch shape - Domain A (night): torch.Size([4, 3, 256, 256])
Batch shape - Domain B (day): torch.Size([4, 3, 256, 256])
✅ Dataloader test successful!
```
→ **Parfait ! Passer au test suivant.**

#### Cas 2 : ❌ Erreur de connexion
```
ConnectionError: Unable to download dataset
```

**Solutions :**
1. Vérifier connexion internet
2. Login Hugging Face : `huggingface-cli login`
3. Télécharger manuellement :
   ```python
   # Dans unpaired.py, modifier use_huggingface=False
   # et télécharger depuis : https://huggingface.co/datasets/huggan/night2day
   ```

#### Cas 3 : ❌ Structure dataset différente
```
KeyError: 'train' or KeyError: 'image'
```

**Solution : Adapter le dataloader**

D'abord, inspecter le dataset :
```python
from datasets import load_dataset
dataset = load_dataset("huggan/night2day")
print(dataset)  # Voir les splits disponibles
print(dataset.keys())  # Voir les clés
if 'train' in dataset:
    print(dataset['train'][0])  # Voir un exemple
    print(dataset['train'][0].keys())  # Voir les champs
```

Ensuite, adapter `src/datasets/unpaired.py` lignes 65-75 :

```python
# AVANT (supposé)
if mode == 'train':
    self.images_a = dataset['train']
    self.images_b = dataset['train']
else:
    self.images_a = dataset['test']
    self.images_b = dataset['test']

# APRÈS (exemple si structure différente)
# Adapter selon ce que print() a montré
```

#### Cas 4 : ❌ Format d'image différent
```
AttributeError: 'dict' object has no attribute 'image'
```

**Solution :** Modifier ligne 102 de `unpaired.py` :
```python
# Inspecter d'abord ce qui est retourné
print(self.images_a[idx])  # Voir la structure

# Adapter selon le format
# Si c'est {'img': PIL.Image} au lieu de {'image': PIL.Image}
img_a = self.images_a[idx]['img']  # au lieu de 'image'
```

---

### ✅ TEST 2: Modèle (2 min)

```bash
python src/models/gan_generator.py
```

#### Cas 1 : ✅ Succès
```
Generator input shape: torch.Size([2, 3, 256, 256])
Generator output shape: torch.Size([2, 3, 256, 256])
Output range: [0.000, 1.000]
Generator parameters: 7,123,456 (~7.12M)
Discriminator output shape: torch.Size([2, 1, 30, 30])
✅ Model test successful!
```
→ **Parfait ! Noter la shape du discriminator.**

#### Cas 2 : ❌ Erreur de dimension
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Solution :** Problème rare, vérifier PyTorch version :
```bash
pip install --upgrade torch torchvision
```

---

### ✅ TEST 3: Training (1 epoch = 10-20 min)

**Objectif :** Vérifier qu'un epoch complet fonctionne

**Modifier temporairement la config :**
```yaml
# Dans src/configs/night2day.yaml
training:
  epochs: 1  # ← Mettre 1 au lieu de 100
  batch_size: 2  # ← Réduire si petit GPU
```

**Lancer :**
```bash
python src/train.py --config src/configs/night2day.yaml
```

#### Cas 1 : ✅ Succès
```
Starting training for 1 epochs
Device: cuda
Batch size: 2

Epoch 0: 100%|████████| 250/250 [02:34<00:00]
  G_total: 12.3456
  G_gan: 0.8234
  G_cycle: 1.1234
  D_A: 0.4567
  D_B: 0.4321
✓ Best model saved at epoch 0
✅ Training complete!
```
→ **Excellent ! Remettre epochs: 100 et lancer vraiment.**

#### Cas 2 : ❌ CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions (par ordre de préférence) :**

1. **Réduire batch_size :**
   ```yaml
   batch_size: 2  # ou même 1
   ```

2. **Réduire image_size :**
   ```yaml
   image_size: 192  # au lieu de 256
   ```

3. **Utiliser modèle lite :**
   ```yaml
   model:
     generator:
       base_filters: 32  # au lieu de 64
   ```

4. **Désactiver mixed precision (contre-intuitif mais parfois aide) :**
   ```yaml
   optimization:
     mixed_precision: false
   ```

#### Cas 3 : ❌ Erreur de dimension dans le discriminator
```
RuntimeError: The size of tensor a (30) must match the size of tensor b (16)
```

**Problème :** La shape du discriminator output ne correspond pas aux labels.

**Solution :** Adapter les dimensions dans `src/train.py` lignes 138-139 :

```python
# AVANT (valeurs supposées)
valid = torch.ones(batch_size, 1, 30, 30, device=self.device)
fake = torch.zeros(batch_size, 1, 30, 30, device=self.device)

# APRÈS : Calculer dynamiquement
# D'abord, tester le discriminator avec un batch :
with torch.no_grad():
    test_input = torch.randn(1, 3, 256, 256).to(self.device)
    test_output = self.disc_A(test_input)
    print(f"Discriminator output shape: {test_output.shape}")
    # Utiliser cette shape pour valid/fake

# Exemple si output est [1, 1, 16, 16] :
valid = torch.ones(batch_size, 1, 16, 16, device=self.device)
fake = torch.zeros(batch_size, 1, 16, 16, device=self.device)
```

**OU automatiser :**
```python
# Remplacer les lignes 138-139 par :
# Calculer la shape dynamiquement
with torch.no_grad():
    dummy_output = self.disc_A(real_A[:1])
    disc_shape = dummy_output.shape[2:]  # (H, W)

valid = torch.ones(batch_size, 1, *disc_shape, device=self.device)
fake = torch.zeros(batch_size, 1, *disc_shape, device=self.device)
```

#### Cas 4 : ❌ Losses explosent (NaN)
```
Epoch 5: G_total: nan, D_A: nan
```

**Causes possibles :**
1. Learning rate trop élevé
2. Discriminateur trop fort
3. Pas de gradient clipping

**Solutions :**

1. **Réduire learning rates :**
   ```yaml
   training:
     lr_generator: 0.0001  # au lieu de 0.0002
     lr_discriminator: 0.0001
   ```

2. **Activer gradient clipping :**
   ```yaml
   optimization:
     clip_grad_norm: 1.0  # Déjà activé normalement
   ```

3. **Ajouter du label smoothing :**
   Dans `train.py`, modifier lignes 138-139 :
   ```python
   # Au lieu de 1.0 et 0.0, utiliser :
   valid = torch.ones(batch_size, 1, H, W, device=self.device) * 0.9
   fake = torch.zeros(batch_size, 1, H, W, device=self.device) * 0.1
   ```

---

### ✅ TEST 4: Evaluation (5 min)

**Après avoir au moins 1 checkpoint** (même avec 1 epoch) :

```bash
python src/eval.py --config src/configs/night2day.yaml \
                    --weights src/runs/night2day_cyclegan/best.pt
```

#### Cas 1 : ✅ Succès
```
Model loaded from src/runs/night2day_cyclegan/best.pt
Direction: AtoB (Night→Day)

Evaluating on 100 test images...
Generating samples: 100%|████| 16/16

✅ Saved 16 sample images to src/runs/night2day_cyclegan/samples/
Metadata saved to src/runs/night2day_cyclegan/test_metrics.json
```
→ **Parfait ! Vérifier visuellement les samples.**

#### Cas 2 : ❌ Checkpoint introuvable
```
FileNotFoundError: src/runs/night2day_cyclegan/best.pt
```

**Solution :** Vérifier que le training a bien sauvegardé :
```bash
ls -la src/runs/night2day_cyclegan/
# Doit contenir best.pt et/ou last.pt
```

Si `last.pt` existe mais pas `best.pt`, utiliser :
```bash
python src/eval.py --config ... --weights src/runs/night2day_cyclegan/last.pt
```

---

### ✅ TEST 5: Demo Jetson (Sur Jetson uniquement)

**Setup Jetson :**
```bash
# 1. Copier le projet
scp -r night2day_project/ jetson@<IP>:~/

# 2. Copier le checkpoint
scp src/runs/night2day_cyclegan/best.pt jetson@<IP>:~/night2day_project/src/runs/night2day_cyclegan/

# 3. SSH sur Jetson
ssh jetson@<IP>
cd ~/night2day_project
```

**Installer dépendances (si pas déjà fait) :**
```bash
pip3 install -r requirements.txt
```

**Tester :**
```bash
python3 src/demo_live_split.py --weights src/runs/night2day_cyclegan/best.pt --size 256
```

#### Cas 1 : ✅ Succès
```
Model loaded: AtoB (Night→Day)
Camera initialized

=== Live Demo Started ===
Press 'q' to quit
Press 's' to save current frame

[Fenêtre OpenCV avec split-screen apparaît]
FPS: 12.3
```
→ **Parfait !**

#### Cas 2 : ❌ Caméra CSI ne s'ouvre pas
```
CSI camera not available, trying USB camera...
Cannot open camera!
```

**Solutions :**

1. **Vérifier caméra CSI :**
   ```bash
   ls /dev/video*
   v4l2-ctl --list-devices
   ```

2. **Tester avec nvgstcapture :**
   ```bash
   nvgstcapture-1.0
   # Si ça marche, la caméra est OK
   ```

3. **Adapter le pipeline GStreamer :**
   Dans `demo_live_split.py`, ligne 88-95, essayer :
   ```python
   # Version simplifiée
   gst_pipeline = (
       "nvarguscamerasrc ! "
       "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
       "nvvidconv ! video/x-raw, format=BGRx ! "
       "videoconvert ! appsink"
   )
   ```

4. **Utiliser USB camera :**
   ```python
   # Forcer USB (ligne 100)
   self.cap = cv2.VideoCapture(0)
   ```

#### Cas 3 : ❌ FPS trop faible (< 5)
```
FPS: 3.2
```

**Solutions (par ordre) :**

1. **Réduire résolution :**
   ```bash
   python3 src/demo_live_split.py --weights best.pt --size 192
   # ou --size 128
   ```

2. **Export ONNX :**
   ```bash
   python3 src/scripts/export_onnx.py --weights best.pt --out model.onnx
   # Puis adapter demo pour charger ONNX
   ```

3. **TensorRT (avancé) :**
   ```bash
   trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
   ```

---

## 🐛 PROBLÈMES FRÉQUENTS ET SOLUTIONS

### Problème 1 : "Module not found"
```python
ModuleNotFoundError: No module named 'datasets'
```

**Solution :**
```bash
pip install datasets huggingface-hub
```

### Problème 2 : DataLoader workers crash (Windows)
```
RuntimeError: DataLoader worker (pid XXXX) exited unexpectedly
```

**Solution :**
```yaml
# Dans night2day.yaml
hardware:
  num_workers: 0  # ← Mettre à 0 sur Windows
```

### Problème 3 : Training très lent sans GPU
**Si pas de CUDA/MPS disponible :**

1. **Accepter CPU (très lent)** :
   ```yaml
   hardware:
     device: "cpu"
   training:
     epochs: 20  # Réduire drastiquement
     batch_size: 1
     image_size: 128  # Réduire résolution
   ```

2. **Ou utiliser Google Colab :**
   - Upload le projet sur Colab
   - GPU gratuit K80/T4

### Problème 4 : Les images générées sont identiques (mode collapse)
**Symptôme :** Le générateur produit toujours la même image

**Solutions :**

1. **Augmenter lambda_cycle :**
   ```yaml
   training:
     lambda_cycle: 15.0  # au lieu de 10.0
   ```

2. **Réduire lr du discriminateur :**
   ```yaml
   training:
     lr_discriminator: 0.0001  # au lieu de 0.0002
   ```

3. **Restart training avec seed différent**

### Problème 5 : Import errors avec Python paths
```
ModuleNotFoundError: No module named 'datasets.unpaired'
```

**Solution :**
```python
# Ajouter au début des scripts train.py, eval.py, etc :
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

---

## 📊 MÉTRIQUES DE RÉFÉRENCE

### Training (après 50-100 epochs)

**Losses attendues :**
- `G_total`: 3-6 (convergence)
- `G_gan`: 0.5-2.0
- `G_cycle`: 0.5-1.5
- `D_A, D_B`: 0.3-0.8

**⚠️ Signes de problème :**
- `G_total > 20` après 20 epochs → LR trop élevé
- `G_gan < 0.1` → Discriminateur trop fort
- `NaN` → Explosion de gradient
- Losses stables mais images mauvaises → Mode collapse

### Performance Jetson

**FPS attendu (PyTorch, FP16) :**
- Jetson Nano 4GB @ 128px: ~8-12 FPS
- Jetson Nano 4GB @ 256px: ~4-6 FPS
- Xavier NX @ 256px: ~15-20 FPS
- Orin Nano @ 256px: ~25-30 FPS

**Si FPS < attendu de 50%+ :**
→ Vérifier que GPU est utilisé (pas CPU fallback)

---

## ✅ CHECKLIST FINALE AVANT DISTRIBUTION

Avant de donner le projet aux collègues, vérifier :

- [ ] Test 1 (Dataset) passé ✅
- [ ] Test 2 (Modèle) passé ✅
- [ ] Test 3 (Training 1 epoch) passé ✅
- [ ] Adapté les dimensions si nécessaire
- [ ] Documenté les changements dans un fichier `ADAPTATIONS.md`
- [ ] Créé un exemple de checkpoint (même avec 1 epoch)
- [ ] Testé sur Jetson si possible

---

## 📝 TEMPLATE ADAPTATIONS.MD

Si vous devez faire des adaptations, documentez-les :

```markdown
# Adaptations effectuées

## Dataset
- [x] Structure différente de ce qui était supposé
- [x] Modifié `unpaired.py` ligne 70 : `dataset['train']` → `dataset['training']`
- [x] Nombre réel d'images : 1234 par domaine

## Modèle
- [ ] Aucune adaptation nécessaire
- OU
- [x] Discriminator output shape : [B,1,16,16] au lieu de [B,1,30,30]
- [x] Modifié `train.py` lignes 138-139

## Training
- [x] Batch size réduit à 2 (GPU 6GB)
- [x] Image size réduit à 192 (au lieu de 256)

## Jetson
- [x] Caméra USB au lieu de CSI
- [x] FPS obtenu : 8-10 @ 192px (Xavier NX)
```

---

## 🎯 MESSAGE POUR VOS COLLÈGUES

Quand vous leur distribuez, expliquez :

> "Le projet est structuré de manière professionnelle et le code suit les best practices CycleGAN. **Cependant**, comme pour tout projet ML, il faudra probablement faire quelques ajustements mineurs selon votre environnement :
> 
> 1. La structure exacte du dataset Hugging Face
> 2. Votre GPU (batch size, résolution)
> 3. Votre modèle Jetson
> 
> **Suivez le guide TESTING.md étape par étape.** Les tests initiaux (15-30 min) vous diront exactement ce qui doit être adapté. Le guide contient toutes les solutions."

---

## 💡 CONSEILS FINAUX

1. **Ne paniquez pas si le premier test échoue** - C'est normal en ML
2. **Testez étape par étape** - Ne passez pas au suivant si le précédent échoue
3. **Documentez vos adaptations** - Ça aidera les autres
4. **Commencez petit** - 1 epoch, petit batch, avant le training complet
5. **La qualité visuelle compte plus que les métriques** - C'est unpaired data

---

**Ce guide transforme un "ça ne marche pas !" en "voici comment l'adapter" 🛠️**

_Bon debugging ! 🐛_
