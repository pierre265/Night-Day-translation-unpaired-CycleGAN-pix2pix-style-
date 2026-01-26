#!/usr/bin/env python3
"""
Script utilitaire pour automatiser les tâches courantes du projet
Compatible Windows, Mac, Linux

Usage:
    python run.py [commande]

Commandes disponibles:
    check       - Vérifier l'environnement
    test-data   - Tester le dataloader
    test-model  - Tester le modèle
    train       - Lancer l'entraînement
    eval        - Évaluer le modèle
    demo        - Lancer le demo live
    export      - Exporter en ONNX
    clean       - Nettoyer les fichiers temporaires
"""

import sys
import os
import subprocess
from pathlib import Path

# Configuration
CONFIG_PATH = "src/configs/night2day.yaml"
WEIGHTS_PATH = "src/runs/night2day_cyclegan/best.pt"

def run_command(cmd, description=""):
    """Execute une commande"""
    if description:
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"{'='*60}\n")
    
    print(f"$ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Erreur lors de l'exécution de: {cmd}")
        sys.exit(1)

def check():
    """Vérifier l'environnement"""
    run_command("python check_environment.py", "Vérification de l'environnement")

def test_data():
    """Tester le dataloader"""
    run_command("python src/datasets/unpaired.py", "Test du dataloader")

def test_model():
    """Tester le modèle"""
    run_command("python src/models/gan_generator.py", "Test du modèle")

def train():
    """Lancer l'entraînement"""
    run_command(
        f"python src/train.py --config {CONFIG_PATH}",
        "Entraînement du modèle CycleGAN"
    )

def eval():
    """Évaluer le modèle"""
    if not Path(WEIGHTS_PATH).exists():
        print(f"❌ Checkpoint introuvable: {WEIGHTS_PATH}")
        print("   Lancez d'abord l'entraînement: python run.py train")
        sys.exit(1)
    
    run_command(
        f"python src/eval.py --config {CONFIG_PATH} --weights {WEIGHTS_PATH}",
        "Évaluation du modèle"
    )

def demo(size=256):
    """Lancer le demo live"""
    if not Path(WEIGHTS_PATH).exists():
        print(f"❌ Checkpoint introuvable: {WEIGHTS_PATH}")
        print("   Lancez d'abord l'entraînement: python run.py train")
        sys.exit(1)
    
    run_command(
        f"python src/demo_live_split.py --config {CONFIG_PATH} --weights {WEIGHTS_PATH} --size {size}",
        f"Demo live (résolution {size}x{size})"
    )

def export():
    """Exporter en ONNX"""
    if not Path(WEIGHTS_PATH).exists():
        print(f"❌ Checkpoint introuvable: {WEIGHTS_PATH}")
        sys.exit(1)
    
    output_path = "src/runs/night2day_cyclegan/model.onnx"
    run_command(
        f"python src/scripts/export_onnx.py --config {CONFIG_PATH} --weights {WEIGHTS_PATH} --out {output_path}",
        "Export ONNX"
    )

def clean():
    """Nettoyer les fichiers temporaires"""
    print("🧹 Nettoyage des fichiers temporaires...")
    
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/.DS_Store",
        "**/*.swp",
        "**/*.tmp"
    ]
    
    count = 0
    for pattern in patterns:
        for path in Path('.').glob(pattern):
            if path.is_file():
                path.unlink()
                count += 1
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
                count += 1
    
    print(f"✅ {count} fichiers/dossiers supprimés")

def help():
    """Afficher l'aide"""
    print(__doc__)

def main():
    if len(sys.argv) < 2:
        help()
        return
    
    command = sys.argv[1]
    
    commands = {
        'check': check,
        'test-data': test_data,
        'test-model': test_model,
        'train': train,
        'eval': eval,
        'demo': demo,
        'export': export,
        'clean': clean,
        'help': help,
        '--help': help,
        '-h': help
    }
    
    if command not in commands:
        print(f"❌ Commande inconnue: {command}")
        print("\nCommandes disponibles:")
        for cmd in commands:
            if not cmd.startswith('-'):
                print(f"  - {cmd}")
        sys.exit(1)
    
    # Exécuter la commande
    try:
        commands[command]()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
