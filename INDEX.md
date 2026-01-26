# 📦 INDEX - Night→Day Translation Project

**Projet complet préparé pour:** ESIEA Embedded AI - Projet 9  
**Préparé le:** 2026-01-26  
**Dataset:** huggan/night2day (Hugging Face)  
**Modèle:** CycleGAN (unpaired image-to-image translation)

---

## 📂 Structure du projet (19 fichiers)

```
night2day_project/
├── 📘 Documentation (7 fichiers)
│   ├── README.md                    ⭐ Documentation principale (15+ pages)
│   ├── QUICKSTART.md                ⚡ Installation rapide (5 min)
│   ├── TODO.md                      📋 Checklist des tâches
│   ├── INSTRUCTOR_NOTES.md          🎓 Notes pour l'enseignant
│   ├── report.md                    📝 Template de rapport final
│   ├── STRUCTURE.txt                📂 Arborescence du projet
│   └── INDEX.md                     📖 Ce fichier
│
├── 🔧 Configuration (2 fichiers)
│   ├── requirements.txt             📦 Dépendances Python
│   └── .gitignore                   🚫 Git ignore
│
├── 🛠️ Scripts utilitaires (2 fichiers)
│   ├── check_environment.py         ✅ Vérification environnement
│   └── run.py                       🚀 Automation des commandes
│
├── 📊 Dataset (1 fichier)
│   └── data/
│       └── DATASET_NOTES.txt        📄 Documentation dataset
│
└── 💻 Code source (7 fichiers Python)
    └── src/
        ├── configs/
        │   └── night2day.yaml       ⚙️  Configuration CycleGAN
        ├── datasets/
        │   ├── __init__.py
        │   └── unpaired.py          📥 Dataloader HuggingFace
        ├── models/
        │   ├── __init__.py
        │   └── gan_generator.py     🧠 CycleGAN (Generator + Discriminator)
        ├── scripts/
        │   └── export_onnx.py       📤 Export ONNX
        ├── train.py                 🎯 Script d'entraînement
        ├── eval.py                  📈 Script d'évaluation
        └── demo_live_split.py       🎥 Demo temps-réel Jetson
```

**Total:** ~2500 lignes de code + 6000+ mots de documentation

---

## 🚀 Démarrage rapide

### 1. Installation (5 min)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Mac
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt

# Vérifier
python check_environment.py
```

### 2. Test (2 min)

```bash
# Tester dataloader
python src/datasets/unpaired.py

# Tester modèle
python src/models/gan_generator.py
```

### 3. Training (4-12h selon GPU)

```bash
# Lancer entraînement
python src/train.py --config src/configs/night2day.yaml
# OU
python run.py train
```

### 4. Evaluation (5 min)

```bash
# Générer samples
python src/eval.py --config src/configs/night2day.yaml \
                    --weights src/runs/night2day_cyclegan/best.pt
# OU
python run.py eval
```

### 5. Demo Jetson (temps-réel)

```bash
# Sur Jetson
python3 src/demo_live_split.py --weights best.pt --size 256
# OU
python3 run.py demo
```

---

## 📚 Guides par rôle

### Pour les étudiants 👨‍🎓

**Commencer ici:**
1. 📘 `QUICKSTART.md` - Installation
2. 📋 `TODO.md` - Liste des tâches
3. 📘 `README.md` - Documentation complète

**En cas de problème:**
- Section "Troubleshooting" du README
- `check_environment.py` pour vérifier le setup
- Commentaires dans le code

### Pour l'enseignant 👨‍🏫

**Lire d'abord:**
1. 🎓 `INSTRUCTOR_NOTES.md` - Notes pédagogiques
2. 📘 `README.md` - Vue d'ensemble complète
3. 📊 `data/DATASET_NOTES.txt` - Détails dataset

**Points d'évaluation:**
- Code: 30% (dataloader, training, demo)
- Résultats: 40% (qualité, FPS, samples)
- Rapport: 30% (documentation, analyse, lessons)

---

## 🎯 Objectifs du projet

1. ✅ Comprendre CycleGAN et cycle consistency
2. ✅ Manipuler datasets unpaired (Hugging Face)
3. ✅ Entraîner modèle adversarial stable
4. ✅ Évaluer qualitativement (pas de PSNR)
5. ✅ Optimiser pour temps-réel Jetson
6. ✅ Analyser trade-offs accuracy vs speed

---

## 📊 Résultats attendus

| Métrique | Valeur cible | Note |
|----------|--------------|------|
| **Epochs training** | 50-100 | CycleGAN nécessite plus d'epochs |
| **G_total loss** | < 5.0 (final) | Somme des losses |
| **Cycle loss** | < 1.0 (final) | Cohérence A→B→A |
| **FPS Jetson** | ≥ 10 @ 256px | Jetson Xavier NX recommandé |
| **Model size** | ~15-20MB | ResNet-6 blocks |
| **Samples test** | 8-16 images | Évaluation qualitative |

---

## ⚙️ Configuration recommandée

### Pour training (Phase 2)

**GPU recommandé:**
- RTX 3060 12GB (ou mieux)
- Batch size: 4-8
- Temps: 4-6h

**GPU minimum:**
- GTX 1060 6GB
- Batch size: 2
- Temps: 8-12h

**Sans GPU:**
- Possible mais très lent (50-100h)
- Réduire epochs à 20-30

### Pour demo (Phase 3)

**Jetson recommandé:**
- Xavier NX (8GB): 15-20 FPS @ 256px
- Orin Nano: 25-30 FPS @ 256px

**Jetson minimum:**
- Nano (4GB): 8-10 FPS @ 128-192px

---

## 📦 Fichiers clés

### Code (à exécuter)

| Fichier | Description | Commande |
|---------|-------------|----------|
| `src/datasets/unpaired.py` | Dataloader | `python src/datasets/unpaired.py` |
| `src/models/gan_generator.py` | Modèle CycleGAN | `python src/models/gan_generator.py` |
| `src/train.py` | Entraînement | `python src/train.py --config ...` |
| `src/eval.py` | Évaluation | `python src/eval.py --config ... --weights ...` |
| `src/demo_live_split.py` | Demo Jetson | `python src/demo_live_split.py --weights ...` |
| `src/scripts/export_onnx.py` | Export ONNX | `python src/scripts/export_onnx.py ...` |

### Configuration (à modifier si besoin)

| Fichier | Description | À ajuster |
|---------|-------------|-----------|
| `src/configs/night2day.yaml` | Config principale | batch_size, epochs, device |
| `requirements.txt` | Dépendances | Si besoin packages supplémentaires |

### Documentation (à lire)

| Fichier | Public | Contenu |
|---------|--------|---------|
| `README.md` | Tous | Documentation complète (15+ pages) |
| `QUICKSTART.md` | Étudiants | Installation rapide (5 min) |
| `TODO.md` | Étudiants | Checklist des tâches |
| `INSTRUCTOR_NOTES.md` | Enseignant | Notes pédagogiques |
| `report.md` | Étudiants | Template de rapport final |
| `data/DATASET_NOTES.txt` | Tous | Détails du dataset |

---

## 🎓 Livrables finaux attendus

1. ✅ Code fonctionnel (tous les scripts)
2. ✅ Checkpoint `best.pt` (modèle entraîné)
3. ✅ Samples test (8-16 images dans `samples/`)
4. ✅ Rapport `report.md` complété
5. ✅ FPS Jetson mesuré et documenté
6. ✅ Au moins 3 "lessons learned"

---

## ⏱️ Timeline suggéré (2-3 jours)

**Jour 1: Setup + Training**
- Matin: Installation, tests (2-3h)
- Après-midi: Lancer training, commencer rapport (2h)
- Soir: Training continue (overnight)

**Jour 2: Évaluation + Analyse**
- Matin: Analyser résultats, générer samples (2h)
- Après-midi: Compléter rapport, préparer Jetson (3-4h)

**Jour 3: Demo Jetson + Finalisation**
- Matin: Setup Jetson, demo (2-3h)
- Après-midi: Optimisations, finaliser rapport (2-3h)

**Total:** ~16-20h actif (+ training overnight)

---

## 🆘 Support

### Ressources intégrées

- ✅ `check_environment.py` - Diagnostic automatique
- ✅ `run.py` - Commandes automatisées
- ✅ Section Troubleshooting du README
- ✅ Commentaires inline dans le code

### Ressources externes

- CycleGAN Paper: https://arxiv.org/abs/1703.10593
- Dataset: https://huggingface.co/datasets/huggan/night2day
- PyTorch Jetson: https://forums.developer.nvidia.com/t/pytorch-for-jetson/

---

## ✅ Checklist avant utilisation

- [ ] Tous les fichiers présents (19 fichiers)
- [ ] Python 3.8+ installé
- [ ] GPU avec CUDA (recommandé) ou MPS (Mac)
- [ ] 10GB+ espace disque libre
- [ ] Connexion internet (téléchargement dataset)
- [ ] Accès Jetson pour Phase 3

---

## 📝 Notes importantes

⚠️ **Dataset unpaired:** Pas de métriques PSNR/SSIM (pas de ground truth)  
⚠️ **GAN training:** Peut être instable, patience requise  
⚠️ **Jetson requis:** Pour Phase 3 (demo temps-réel)  
⚠️ **Temps training:** 4-12h selon GPU, prévoir overnight  

✅ **Code complet:** Prêt à l'emploi, pas besoin de code supplémentaire  
✅ **Documentation exhaustive:** 6000+ mots  
✅ **Support multi-OS:** Windows, Mac, Linux  

---

## 🏆 Points bonus (optionnels)

- [ ] Export ONNX + benchmark vitesse
- [ ] Calcul FID/LPIPS (métriques avancées)
- [ ] Fine-tuning sur subset spécifique
- [ ] TensorRT implementation (Jetson)
- [ ] Vidéo demo professionnelle

---

**Le projet est complet et prêt à être utilisé ! 🚀**

_Préparé avec soin pour ESIEA Embedded AI - Bonne chance ! 🎯_
