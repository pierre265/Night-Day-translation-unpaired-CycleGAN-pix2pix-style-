# ⚠️ POINTS D'ATTENTION

**Ce projet est une base solide mais nécessitera probablement quelques ajustements.**

## 🔴 À tester OBLIGATOIREMENT avant de commencer

### 1. Dataset (le plus critique)
```bash
python src/datasets/unpaired.py
```
**Problème probable :** Structure du dataset `huggan/night2day` peut différer  
**Solution :** Adapter `unpaired.py` lignes 65-75 selon la vraie structure

### 2. Training (1 epoch de test)
```bash
# Mettre epochs: 1 dans night2day.yaml
python src/train.py --config src/configs/night2day.yaml
```
**Problèmes probables :**
- OOM → Réduire `batch_size` à 2 ou 1
- Erreur de dimension discriminator → Ajuster lignes 138-139 de `train.py`

### 3. Demo Jetson
```bash
python3 src/demo_live_split.py --weights best.pt --size 256
```
**Problèmes probables :**
- Caméra CSI ne s'ouvre pas → Adapter le pipeline GStreamer
- FPS trop faible → Réduire résolution (`--size 192` ou `128`)

## ⚠️ Ce qui est supposé (non testé)

- ✅ **Code** : Architecture correcte, logique solide
- ⚠️ **Dataset** : Structure supposée, à vérifier
- ⚠️ **Dimensions** : Discriminator output (30x30) estimé
- ⚠️ **Hyperparams** : Standards mais à ajuster selon GPU
- ⚠️ **Performance** : FPS Jetson estimés (±50%)

## ✅ Ce qui marchera sûrement

- Structure du projet
- Imports et dépendances
- Architecture CycleGAN
- Documentation et guides
- Scripts utilitaires

## 🛠️ En cas de problème

Voir le fichier `TESTING.md` (guide complet de debugging)

## 💡 Conseil

C'est normal de devoir adapter du code en ML. Faites les tests, documentez vos changements, et vous serez bons ! 🚀
