# Rapport Final - Night→Day Translation (Projet 9)

**Projet:** CycleGAN pour translation night→day sur Jetson  
**Dataset:** huggan/night2day (Hugging Face)  
**Équipe:** [Noms des membres]  
**Date:** [Date de soumission]

---

## 1. Dataset et Preprocessing

### 1.1 Source du dataset

- **Nom:** huggan/night2day
- **Type:** Unpaired image-to-image translation
- **URL:** https://huggingface.co/datasets/huggan/night2day
- **License:** [À vérifier sur HF]

### 1.2 Caractéristiques

| Attribut | Valeur |
|----------|--------|
| **Domain A** | Images nocturnes (night) |
| **Domain B** | Images diurnes (day) |
| **Train images** | ~[nombre] par domaine |
| **Test images** | ~[nombre] par domaine |
| **Format** | RGB, tailles variables |
| **Appaiement** | ❌ Non (unpaired) |

### 1.3 Preprocessing appliqué

**Transformations train:**
- Resize à `[taille]x[taille]` pixels
- Random horizontal flip (p=0.5)
- Random crop si image > train_size
- Normalisation: [0, 1] range

**Transformations test:**
- Resize à `[taille]x[taille]` pixels
- Pas d'augmentation

**Justification:**
- Pas de normalisation ImageNet pour préserver la cohérence visuelle
- Pas de color jitter (important pour ne pas mélanger les domaines)

### 1.4 Statistiques du dataset

[Ajouter ici des statistiques si calculées: distribution couleurs, résolutions, etc.]

---

## 2. Modèle et Architecture

### 2.1 Choix du modèle

**Modèle sélectionné:** CycleGAN avec générateurs ResNet-[6/9] blocks

**Justification:**
- Adapté pour unpaired data (pas besoin d'images alignées)
- Cycle consistency loss assure cohérence A→B→A
- Architecture éprouvée pour domain translation

### 2.2 Architecture détaillée

**Générateur (ResNet-based):**
```
Input (3 channels) 
  → Encoder: Conv blocks with downsampling
  → Transformation: [6/9] Residual blocks
  → Decoder: Deconv blocks with upsampling
  → Output (3 channels, Tanh)

Parameters: ~[nombre]M
```

**Discriminateur (PatchGAN):**
```
Input (3 channels)
  → [3] Conv blocks with stride 2
  → Output: Patch predictions [B, 1, 30, 30]

Parameters: ~[nombre]M
```

### 2.3 Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| **Epochs** | [100] | CycleGAN nécessite plus d'epochs |
| **Batch size** | [4] | Limitation GPU mémoire |
| **Image size** | [256x256] | Compromis qualité/vitesse |
| **LR (Generator)** | [0.0002] | Standard Adam LR |
| **LR (Discriminator)** | [0.0002] | Même LR pour stabilité |
| **Lambda cycle** | [10.0] | Poids cycle consistency |
| **Lambda identity** | [5.0] | Optionnel, réduit color shift |
| **Optimizer** | Adam (β1=0.5, β2=0.999) | Standard pour GAN |

### 2.4 Loss functions

**GAN Loss (LSGAN):**
```
L_gan = MSE(D(fake), 1) pour Generator
      + MSE(D(real), 1) + MSE(D(fake), 0) pour Discriminator
```

**Cycle Consistency Loss:**
```
L_cycle = |G_BtoA(G_AtoB(A)) - A| + |G_AtoB(G_BtoA(B)) - B|
```

**Identity Loss (optionnel):**
```
L_identity = |G_AtoB(B) - B| + |G_BtoA(A) - A|
```

**Total Loss:**
```
L_total = L_gan + λ_cycle * L_cycle + λ_identity * L_identity
```

---

## 3. Résultats d'entraînement

### 3.1 Métriques (losses)

[Compléter avec les résultats d'entraînement]

| Epoch | G_total | G_gan | G_cycle | D_A | D_B |
|-------|---------|-------|---------|-----|-----|
| 10    | [X.XX]  | [X.XX]| [X.XX]  |[X.XX]|[X.XX]|
| 50    | [X.XX]  | [X.XX]| [X.XX]  |[X.XX]|[X.XX]|
| 100   | [X.XX]  | [X.XX]| [X.XX]  |[X.XX]|[X.XX]|

**Observations:**
- [Convergence atteinte à epoch X]
- [Stabilité du training: oui/non]
- [Problèmes rencontrés: mode collapse, etc.]

### 3.2 Courbes de loss

[Insérer ici les courbes de loss si générées]

```
[Image: loss_curves.png]
```

### 3.3 Samples visuels

**Échantillons train:**

[Insérer 4-6 exemples: input | generated | (optionnel) reconstructed]

**Échantillons test:**

[Insérer 8-16 exemples de `src/runs/night2day_cyclegan/samples/`]

**Analyse qualitative:**
- ✅ **Points forts:** [Ex: couleurs réalistes, cohérence globale]
- ⚠️ **Points faibles:** [Ex: artefacts sur ciel, détails perdus]

---

## 4. Démo Jetson (Phase 3)

### 4.1 Configuration matérielle

**Jetson:**
- Modèle: [Jetson Nano / Xavier NX / Orin Nano]
- RAM: [4GB / 8GB]
- CUDA: [version]

**Caméra:**
- Type: [CSI IMX219 / USB]
- Résolution: [1280x720 @ 30fps]

### 4.2 Performance temps-réel

| Configuration | Résolution | FPS | Latence | Qualité |
|---------------|------------|-----|---------|---------|
| **PyTorch FP32** | 256x256 | [X.X] | [XXms] | ⭐⭐⭐ |
| **PyTorch FP16** | 256x256 | [X.X] | [XXms] | ⭐⭐⭐ |
| **ONNX** | 256x256 | [X.X] | [XXms] | ⭐⭐⭐ |
| **TensorRT (optionnel)** | 256x256 | [X.X] | [XXms] | ⭐⭐⭐ |

**FPS cible:** ≥ 10 FPS @ 256x256

### 4.3 Optimisations appliquées

- [x] Mixed precision (FP16)
- [x] Résolution réduite (256 → [192/128] si nécessaire)
- [ ] Export ONNX
- [ ] TensorRT engine
- [x] Batch size = 1 (inference)

### 4.4 Description du démo

**Affichage split-screen:**
```
┌─────────────┬─────────────┐
│  Original   │  Night→Day  │
│  (camera)   │  (generated)│
│             │             │
│ FPS: XX.X   │             │
└─────────────┴─────────────┘
```

**Fonctionnalités:**
- [x] Capture temps-réel depuis caméra CSI/USB
- [x] Inférence avec générateur A→B (night→day)
- [x] Affichage FPS overlay
- [x] Sauvegarde frames (touche 's')
- [x] Exit propre (touche 'q')

**Observations en conditions réelles:**
- [Tester en environnement nocturne/faible luminosité]
- [Qualité perçue: bonne/moyenne/faible]
- [Stabilité: stable/flickering/artefacts]

---

## 5. Lessons Learned

### 5.1 Trade-offs Accuracy vs Speed

**Dilemme résolution:**
- ⬆️ 320x320 → Meilleure qualité, **mais** FPS < 10 ❌
- ⬇️ 192x192 → FPS > 15 ✅, **mais** détails perdus ⚠️
- ✅ **Choix final:** [256x256 @ XX FPS]

**Dilemme architecture:**
- ResNet-9 blocks → Meilleure qualité, **mais** 2x plus lent
- ResNet-6 blocks → Compromis acceptable ✅
- U-Net lite → Plus rapide, **mais** résultats moins convaincants

### 5.2 Difficultés rencontrées

1. **[Difficulté 1]**
   - Problème: [Description]
   - Solution: [Comment résolu]

2. **[Difficulté 2]**
   - Problème: [Ex: OOM sur Jetson avec batch_size=4]
   - Solution: [batch_size=1 pour inference]

3. **[Difficulté 3]**
   - Problème: [Ex: Dataset unpaired → pas de métrique quantitative]
   - Solution: [Évaluation qualitative uniquement]

### 5.3 Améliorations futures

- [ ] Tester ResNet-9 blocks pour meilleure qualité
- [ ] Implémenter TensorRT pour accélération
- [ ] Ajouter calcul FID/LPIPS pour métrique quantitative (optionnel)
- [ ] Tester sur d'autres datasets (alderley, bdd100k night/day)
- [ ] Fine-tuning sur scènes spécifiques (urbain, rural, etc.)

---

## 6. Conclusion

**Résumé:**

Ce projet a permis d'implémenter avec succès un système de **translation night→day** utilisant CycleGAN, déployé en temps-réel sur NVIDIA Jetson. Les résultats visuels sont [qualifier: convaincants/satisfaisants/acceptables], avec un FPS de [X.X] à résolution 256x256.

**Points clés:**
- ✅ Dataset unpaired accessible via Hugging Face
- ✅ CycleGAN entraîné en [X] epochs
- ✅ Demo temps-réel fonctionnelle sur Jetson
- ⚠️ Évaluation qualitative uniquement (unpaired data)

**Recommandations:**

Pour usage production:
1. Fine-tuning sur données cibles spécifiques
2. Optimisation TensorRT pour FPS > 20
3. Post-processing pour réduire artefacts

---

## 7. Références

- **CycleGAN Paper:** Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017
- **Dataset:** https://huggingface.co/datasets/huggan/night2day
- **Code base:** [Lien vers repo si public]

---

**Annexes:**

- [ ] `src/configs/night2day.yaml` - Configuration complète
- [ ] `src/runs/night2day_cyclegan/samples/` - Échantillons test
- [ ] `demo_video.mp4` - Vidéo démo Jetson (optionnel)

---

_Rapport complété le [date] par [noms]_
