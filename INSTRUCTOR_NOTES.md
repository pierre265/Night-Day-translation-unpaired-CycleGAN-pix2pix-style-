# 📚 INSTRUCTOR NOTES - Night→Day Project

Documentation pour l'équipe pédagogique / Instructeur

---

## 📋 Vue d'ensemble du projet

**Projet:** Night→Day Translation (Projet 9 - CycleGAN)  
**Dataset:** huggan/night2day (Hugging Face, unpaired)  
**Objectif:** Translation d'images nocturnes → diurnes avec démo temps-réel sur Jetson

**Niveau de difficulté:** ⭐⭐⭐⭐ (Intermédiaire-Avancé)
- GAN training (instabilité potentielle)
- Unpaired data (pas de métrique quantitative directe)
- Optimisation temps-réel Jetson

---

## 🎯 Objectifs pédagogiques

Les étudiants vont:
1. ✅ Comprendre les GANs et cycle consistency
2. ✅ Manipuler des datasets non-appariés
3. ✅ Entraîner un modèle adversarial stable
4. ✅ Évaluer qualitativement (FID/LPIPS optionnel)
5. ✅ Optimiser pour inférence temps-réel (embedded)
6. ✅ Gérer les trade-offs accuracy vs speed

---

## 📦 Livrables fournis aux étudiants

### Code complet

- [x] `src/datasets/unpaired.py` - Dataloader Hugging Face
- [x] `src/models/gan_generator.py` - CycleGAN (Generator + Discriminator)
- [x] `src/train.py` - Script d'entraînement complet
- [x] `src/eval.py` - Script d'évaluation
- [x] `src/demo_live_split.py` - Demo temps-réel Jetson
- [x] `src/scripts/export_onnx.py` - Export ONNX
- [x] `src/configs/night2day.yaml` - Configuration complète

### Documentation

- [x] `README.md` - Documentation complète (15+ pages)
- [x] `QUICKSTART.md` - Installation rapide
- [x] `TODO.md` - Checklist des tâches
- [x] `data/DATASET_NOTES.txt` - Infos dataset
- [x] `report.md` - Template de rapport

### Utilities

- [x] `requirements.txt` - Dépendances
- [x] `check_environment.py` - Vérification setup
- [x] `run.py` - Automation scripts
- [x] `.gitignore` - Git ignore

**Total:** ~2000 lignes de code Python + 5000+ mots de documentation

---

## ⏱️ Temps estimé

| Phase | Tâche | Temps estimé |
|-------|-------|--------------|
| **Phase 1** | Setup + Dataset | 2-3h |
| **Phase 2** | Training (GPU) | 6-12h (overnight) |
| **Phase 2** | Eval + Analysis | 1-2h |
| **Phase 3** | Jetson Setup | 1-2h |
| **Phase 3** | Demo + Optimization | 2-3h |
| **Phase 4** | Rapport | 2-4h |
| **TOTAL** | | **~18-28h** sur 2-3 jours |

**Note:** Le training peut tourner en background, donc temps actif ~10-12h.

---

## 🔧 Configuration matérielle recommandée

### Pour training (Phase 2)

**Minimum:**
- GPU: GTX 1060 6GB ou équivalent
- RAM: 8GB
- Stockage: 10GB libre
- Temps training: ~8-12h @ batch_size=2

**Recommandé:**
- GPU: RTX 3060 12GB ou mieux
- RAM: 16GB
- Stockage: 20GB libre
- Temps training: ~4-6h @ batch_size=4-8

**Alternatif (CPU/MPS):**
- Possible mais TRÈS lent (50-100h training)
- Réduire epochs à 20-30 pour tests

### Pour démo (Phase 3)

**Jetson requis:**
- Jetson Nano (4GB): Possible @ 128-192px, ~8-10 FPS
- Jetson Xavier NX (8GB): Recommandé @ 256px, ~15-20 FPS
- Jetson Orin Nano: Optimal @ 256-320px, ~25-30 FPS

---

## 📊 Résultats attendus

### Métriques training

| Epoch | G_total | G_cycle | FPS Jetson @256 |
|-------|---------|---------|-----------------|
| 10    | ~8-12   | ~2-3    | - |
| 50    | ~4-6    | ~1-2    | 10-15 |
| 100   | ~3-5    | ~0.8-1.2| 12-18 |

**Note:** Valeurs indicatives, peuvent varier selon initialisation.

### Qualité visuelle

**Attendu:**
- ✅ Changement nuit→jour visible (couleurs)
- ✅ Cohérence structurelle globale
- ⚠️ Possibles artefacts (grille, bruit) - normal pour CycleGAN
- ⚠️ Manque de détails fins - limitation résolution 256

**Échec:**
- ❌ Mode collapse (génère toujours la même image)
- ❌ Pas de changement visible
- ❌ Artefacts majeurs (corruption totale)

---

## ⚠️ Difficultés attendues

### 1. Training instable

**Symptômes:**
- Losses oscillent violemment
- Mode collapse
- Discriminateur trop fort (G_gan très élevé)

**Solutions proposées:**
- Ajuster learning rates (réduire lr_disc)
- Augmenter lambda_cycle
- Ajouter label smoothing (optionnel)
- Réduire n_layers discriminator

### 2. OOM (Out of Memory)

**Solutions:**
- Réduire batch_size à 2 ou 1
- Réduire image_size à 192 ou 128
- Utiliser gradient accumulation
- Réduire base_filters à 32 (lite model)

### 3. Dataset lent à charger

**Solutions:**
- Premier chargement peut prendre 5-10min (cache ensuite)
- `num_workers=0` sur Windows si crash
- Télécharger dataset manuellement si problème réseau

### 4. FPS Jetson trop faible

**Solutions:**
- Réduire résolution (256 → 192 → 128)
- Export ONNX
- TensorRT (avancé)
- Modèle lite (base_filters=32, 6 blocks)

---

## 🎓 Points d'évaluation suggérés

### Code (30%)
- [ ] Dataloader fonctionnel (10%)
- [ ] Training convergence (10%)
- [ ] Demo Jetson fonctionnel (10%)

### Résultats (40%)
- [ ] Qualité visuelle acceptable (20%)
- [ ] FPS ≥ 10 sur Jetson @ 256px (10%)
- [ ] 8-16 samples test générés (10%)

### Rapport (30%)
- [ ] Documentation complète (10%)
- [ ] Analyse des résultats (10%)
- [ ] Lessons learned pertinents (10%)

**Bonus (+10%):**
- Export ONNX + comparaison vitesse
- Calcul FID/LPIPS
- Fine-tuning sur subset spécifique
- TensorRT implementation

---

## 🔍 Points de vigilance pour l'instructeur

### À vérifier lors des reviews

1. **Dataset:**
   - [ ] Les étudiants utilisent bien HuggingFace (pas de download manuel)
   - [ ] Transformations appropriées (pas de color jitter)

2. **Training:**
   - [ ] Losses diminuent (pas d'explosion)
   - [ ] Checkpoints sauvegardés correctement
   - [ ] Mixed precision activé (si GPU compatible)

3. **Evaluation:**
   - [ ] Évaluation qualitative (pas de fausses métriques PSNR)
   - [ ] Samples visuels diversifiés

4. **Jetson:**
   - [ ] FPS réaliste et mesuré correctement
   - [ ] Split-screen fonctionnel
   - [ ] Résolution documentée

5. **Rapport:**
   - [ ] Honnêteté sur les limitations
   - [ ] Lessons learned concrets
   - [ ] Trade-offs discutés

### Erreurs communes à éviter

❌ **Calculer PSNR/SSIM sur unpaired data**
- Explication: Pas de ground truth, métriques non pertinentes

❌ **Réclamer des résultats parfaits**
- CycleGAN a des limitations, c'est OK

❌ **Négliger l'optimisation Jetson**
- Phase 3 est cruciale pour l'embedded AI

❌ **Copier-coller sans comprendre**
- Code fourni est dense, vérifier compréhension

---

## 📚 Ressources complémentaires

### Papers
- **CycleGAN:** https://arxiv.org/abs/1703.10593
- **Pix2Pix:** https://arxiv.org/abs/1611.07004

### Tutorials
- PyTorch GAN Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- Jetson PyTorch: https://forums.developer.nvidia.com/t/pytorch-for-jetson/

### Datasets alternatifs (si problèmes)
- CycleGAN datasets: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
- BDD100K day/night: https://www.bdd100k.com/

---

## 💬 FAQ Instructeur

**Q: Les étudiants peuvent-ils utiliser un autre dataset?**
A: Oui, mais doit être unpaired. Adapter les paths dans le config.

**Q: Combien de temps minimum pour training?**
A: Minimum 30 epochs (~3-4h sur GPU moyen) pour voir des résultats.

**Q: Peut-on faire Phase 3 sans Jetson?**
A: Oui avec webcam USB sur laptop, mais moins pertinent pour "embedded AI".

**Q: Mode collapse, que faire?**
A: Réduire lr_disc, augmenter lambda_cycle, ou restart avec autre seed.

**Q: FPS trop faible sur Jetson Nano?**
A: Normal. Accepter 128px @ 8-10 FPS comme acceptable.

---

## 📧 Support

Pour questions sur le projet:
- Documentation: Voir README.md
- Code: Commentaires inline
- Bugs: Vérifier `check_environment.py` d'abord

---

_Document préparé pour équipe pédagogique - 2026-01-26_
