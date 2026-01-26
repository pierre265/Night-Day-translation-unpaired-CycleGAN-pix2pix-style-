"""
Script d'évaluation pour CycleGAN - Night→Day Translation
Phase 2: Evaluate and test model

Usage:
    python src/eval.py --config src/configs/night2day.yaml --weights src/runs/night2day_cyclegan/best.pt
"""

import os
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.unpaired import get_dataloader
from models.gan_generator import create_generator


class Evaluator:
    """Classe d'évaluation pour CycleGAN"""
    
    def __init__(self, config: Dict, weights_path: str):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Créer dossier de samples
        self.samples_dir = Path(config['checkpointing']['save_dir']) / 'samples'
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger le modèle
        self._load_model(weights_path)
    
    def _load_model(self, weights_path: str):
        """Charge le générateur depuis un checkpoint"""
        cfg_gen = self.config['model']['generator']
        
        # Créer le générateur
        self.generator = create_generator(
            model_type=cfg_gen['name'],
            input_channels=cfg_gen['input_channels'],
            output_channels=cfg_gen['output_channels'],
            base_filters=cfg_gen['base_filters'],
            use_dropout=False  # Pas de dropout en eval
        ).to(self.device)
        
        # Charger les poids
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Déterminer quelle direction (A→B ou B→A)
        direction = self.config['inference']['direction']
        key = 'gen_AtoB' if direction == 'AtoB' else 'gen_BtoA'
        
        self.generator.load_state_dict(checkpoint[key])
        self.generator.eval()
        
        print(f"Model loaded from {weights_path}")
        print(f"Direction: {direction} ({'Night→Day' if direction == 'AtoB' else 'Day→Night'})")
    
    @torch.no_grad()
    def evaluate(self):
        """Évalue le modèle sur le test set"""
        # Créer dataloader test
        test_loader = get_dataloader(
            batch_size=1,  # 1 image à la fois pour sauvegarder
            image_size=self.config['training']['image_size'],
            num_workers=0,
            mode='test',
            use_huggingface=self.config['dataset']['use_huggingface'],
            hf_dataset_name=self.config['dataset']['name']
        )
        
        print(f"\nEvaluating on {len(test_loader)} test images...")
        
        # Pour unpaired data, on ne peut pas calculer PSNR/SSIM
        # On génère juste des samples visuels
        num_samples = min(self.config['validation']['num_val_samples'], len(test_loader))
        
        for i, (img_a, img_b) in enumerate(tqdm(test_loader, desc="Generating samples")):
            if i >= num_samples:
                break
            
            # Sélectionner l'input selon la direction
            direction = self.config['inference']['direction']
            input_img = img_a if direction == 'AtoB' else img_b
            input_img = input_img.to(self.device)
            
            # Générer output
            output_img = self.generator(input_img)
            
            # Créer grille: [input | output]
            grid = torch.cat([input_img, output_img], dim=3)  # Concat horizontalement
            
            # Sauvegarder
            save_path = self.samples_dir / f'sample_{i:04d}.png'
            save_image(grid, save_path, normalize=False)
        
        print(f"\n✅ Saved {num_samples} sample images to {self.samples_dir}")
        
        # Sauvegarder métadonnées
        metadata = {
            'num_samples': num_samples,
            'direction': self.config['inference']['direction'],
            'model': self.config['model']['generator']['name'],
            'image_size': self.config['training']['image_size'],
            'note': 'Unpaired data - qualitative evaluation only'
        }
        
        with open(self.samples_dir.parent / 'test_metrics.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {self.samples_dir.parent / 'test_metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--weights', type=str, required=True, help='Path to checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Evaluate
    evaluator = Evaluator(config, args.weights)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
