# 📋 TODO - Night→Day Project

Liste des tâches pour compléter le projet.

---

## Phase 1: Dataset Preparation ✅ (Préparé)

- [x] Définir le dataset (huggan/night2day)
- [x] Implémenter le dataloader unpaired
- [x] Documenter le dataset (DATASET_NOTES.txt)
- [ ] **VOTRE TÂCHE:** Tester le dataloader
  ```bash
  python src/datasets/unpaired.py
  ```
- [ ] **VOTRE TÂCHE:** Vérifier que le dataset se télécharge correctement

---

## Phase 2: Training & Evaluation ✅ (Code prêt)

### 2.1 Setup

- [ ] **VOTRE TÂCHE:** Installer les dépendances
  ```bash
  pip install -r requirements.txt
  python check_environment.py
  ```

- [ ] **VOTRE TÂCHE:** Ajuster la configuration si nécessaire
  - Fichier: `src/configs/night2day.yaml`
  - Paramètres clés:
    - `batch_size`: 4 (réduire si OOM)
    - `epochs`: 100 (peut être réduit à 50 pour tests)
    - `device`: "cuda" (ou "mps" sur Mac, "cpu" en dernier recours)

### 2.2 Training

- [ ] **VOTRE TÂCHE:** Lancer l'entraînement
  ```bash
  python src/train.py --config src/configs/night2day.yaml
  # OU
  python run.py train
  ```

- [ ] **VOTRE TÂCHE:** Monitorer les losses
  - G_total devrait diminuer progressivement
  - Cycle loss devrait converger vers < 1.0
  - Sauvegardes automatiques tous les 10 epochs

- [ ] **OPTIONNEL:** Ajuster hyperparamètres si nécessaire
  - Learning rate
  - Lambda cycle/identity
  - Nombre de residual blocks

### 2.3 Evaluation

- [ ] **VOTRE TÂCHE:** Évaluer le modèle
  ```bash
  python src/eval.py --config src/configs/night2day.yaml \
                      --weights src/runs/night2day_cyclegan/best.pt
  # OU
  python run.py eval
  ```

- [ ] **VOTRE TÂCHE:** Analyser les samples visuels
  - Vérifier: `src/runs/night2day_cyclegan/samples/`
  - Évaluer qualitativement (couleurs, cohérence, artefacts)

---

## Phase 3: Jetson Demo 🎥 (À faire sur Jetson)

### 3.1 Setup Jetson

- [ ] **VOTRE TÂCHE:** Transférer les fichiers sur Jetson
  ```bash
  # Sur votre PC
  scp -r night2day_project/ jetson@<IP>:~/
  
  # Sur Jetson
  cd ~/night2day_project
  pip3 install -r requirements.txt
  ```

- [ ] **VOTRE TÂCHE:** Copier le checkpoint entraîné
  ```bash
  # Copier best.pt depuis PC vers Jetson
  scp src/runs/night2day_cyclegan/best.pt jetson@<IP>:~/night2day_project/src/runs/night2day_cyclegan/
  ```

### 3.2 Demo temps-réel

- [ ] **VOTRE TÂCHE:** Tester le demo
  ```bash
  python3 src/demo_live_split.py \
      --weights src/runs/night2day_cyclegan/best.pt \
      --size 256
  # OU
  python3 run.py demo
  ```

- [ ] **VOTRE TÂCHE:** Mesurer les performances
  - Noter le FPS affiché
  - Tester différentes résolutions (128, 192, 256, 320)
  - Choisir le meilleur compromis qualité/vitesse

- [ ] **OPTIONNEL:** Optimisations
  - [ ] Export ONNX: `python run.py export`
  - [ ] TensorRT (avancé): Voir README.md

### 3.3 Captures

- [ ] **VOTRE TÂCHE:** Capturer des vidéos/screenshots
  - Utiliser touche 's' dans le demo
  - Enregistrer une courte vidéo du demo (optionnel)

---

## Phase 4: Documentation & Rapport 📝

- [ ] **VOTRE TÂCHE:** Compléter le rapport (`report.md`)
  - [ ] Section 1: Dataset (sources, preprocessing)
  - [ ] Section 2: Modèle (architecture, hyperparamètres)
  - [ ] Section 3: Résultats (losses, samples)
  - [ ] Section 4: Demo Jetson (FPS, optimisations)
  - [ ] Section 5: Lessons learned

- [ ] **VOTRE TÂCHE:** Ajouter les résultats
  - [ ] Courbes de loss (screenshot TensorBoard ou logs)
  - [ ] 8-16 images de test
  - [ ] Tableau de performances Jetson

- [ ] **OPTIONNEL:** Améliorer le README si nécessaire

---

## Checklist Finale 🎯

Avant de soumettre, vérifier que vous avez:

- [ ] Code complet et fonctionnel
- [ ] Checkpoint `best.pt` sauvegardé
- [ ] Samples d'évaluation (8-16 images)
- [ ] Rapport `report.md` complété
- [ ] FPS Jetson mesuré et documenté
- [ ] Au moins 3 "lessons learned"

---

## Timeline Suggéré ⏱️

**Jour 1:**
- Setup environnement
- Tester dataloader et modèle
- Lancer training (laisser tourner overnight si possible)

**Jour 2:**
- Analyser résultats training
- Évaluation et génération samples
- Commencer rapport

**Jour 3 (Jetson):**
- Setup Jetson
- Demo temps-réel
- Optimisations
- Finaliser rapport

---

## Ressources Utiles 📚

- `README.md` - Documentation complète
- `QUICKSTART.md` - Installation rapide
- `data/DATASET_NOTES.txt` - Infos dataset
- `check_environment.py` - Vérifier setup
- `run.py` - Commandes automatisées

---

## Contacts / Questions ❓

Si vous rencontrez des problèmes:

1. Consulter la section Troubleshooting du README
2. Vérifier les commentaires dans le code
3. [Ajouter vos contacts ici]

---

**Bon courage! 🚀**

_Dernière mise à jour: 2026-01-26_
