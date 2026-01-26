"""
Script d'entraînement pour CycleGAN - Night→Day Translation
Phase 2: Train, evaluate, test model

Usage:
    python src/train.py --config src/configs/night2day.yaml
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
from datetime import datetime

# Imports locaux
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets.unpaired import get_dataloader
from models.gan_generator import create_generator, create_discriminator


class CycleGANTrainer:
    """Classe d'entraînement pour CycleGAN"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Créer le dossier de sauvegarde
        self.save_dir = Path(config['checkpointing']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder la config
        with open(self.save_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # Initialiser les modèles
        self._build_models()
        
        # Initialiser les optimizers
        self._build_optimizers()
        
        # Loss functions
        self.criterion_gan = nn.MSELoss()  # LSGAN loss
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Mixed precision
        self.use_amp = config['optimization']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
    def _build_models(self):
        """Initialise les générateurs et discriminateurs"""
        cfg_gen = self.config['model']['generator']
        cfg_disc = self.config['model']['discriminator']
        
        # Générateurs: A→B (night→day) et B→A (day→night)
        self.gen_AtoB = create_generator(
            model_type=cfg_gen['name'],
            input_channels=cfg_gen['input_channels'],
            output_channels=cfg_gen['output_channels'],
            base_filters=cfg_gen['base_filters'],
            use_dropout=cfg_gen['use_dropout']
        ).to(self.device)
        
        self.gen_BtoA = create_generator(
            model_type=cfg_gen['name'],
            input_channels=cfg_gen['input_channels'],
            output_channels=cfg_gen['output_channels'],
            base_filters=cfg_gen['base_filters'],
            use_dropout=cfg_gen['use_dropout']
        ).to(self.device)
        
        # Discriminateurs: pour domaine A et domaine B
        self.disc_A = create_discriminator(
            input_channels=cfg_disc['input_channels'],
            base_filters=cfg_disc['base_filters'],
            n_layers=cfg_disc['n_layers']
        ).to(self.device)
        
        self.disc_B = create_discriminator(
            input_channels=cfg_disc['input_channels'],
            base_filters=cfg_disc['base_filters'],
            n_layers=cfg_disc['n_layers']
        ).to(self.device)
        
        print(f"Models initialized on {self.device}")
    
    def _build_optimizers(self):
        """Initialise les optimizers"""
        lr_gen = self.config['training']['lr_generator']
        lr_disc = self.config['training']['lr_discriminator']
        beta1 = self.config['training']['beta1']
        beta2 = self.config['training']['beta2']
        
        # Optimizer pour les générateurs
        self.optimizer_G = torch.optim.Adam(
            list(self.gen_AtoB.parameters()) + list(self.gen_BtoA.parameters()),
            lr=lr_gen,
            betas=(beta1, beta2)
        )
        
        # Optimizer pour les discriminateurs
        self.optimizer_D = torch.optim.Adam(
            list(self.disc_A.parameters()) + list(self.disc_B.parameters()),
            lr=lr_disc,
            betas=(beta1, beta2)
        )
        
        print("Optimizers initialized")
    
    def train_epoch(self, dataloader):
        """Entraîne une epoch"""
        self.gen_AtoB.train()
        self.gen_BtoA.train()
        self.disc_A.train()
        self.disc_B.train()
        
        epoch_losses = {
            'G_total': 0.0,
            'G_gan': 0.0,
            'G_cycle': 0.0,
            'G_identity': 0.0,
            'D_A': 0.0,
            'D_B': 0.0
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (real_A, real_B) in enumerate(pbar):
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            batch_size = real_A.size(0)
            
            # Labels pour GAN loss
            valid = torch.ones(batch_size, 1, 30, 30, device=self.device)  # PatchGAN output
            fake = torch.zeros(batch_size, 1, 30, 30, device=self.device)
            
            # ================== Train Generators ==================
            self.optimizer_G.zero_grad()
            
            with autocast() if self.use_amp else torch.cuda.amp.autocast(enabled=False):
                # Identity loss (optionnel)
                loss_identity = 0
                lambda_identity = self.config['training']['lambda_identity']
                if lambda_identity > 0:
                    # G_AtoB(B) should equal B
                    identity_B = self.gen_AtoB(real_B)
                    loss_identity_B = self.criterion_identity(identity_B, real_B)
                    # G_BtoA(A) should equal A
                    identity_A = self.gen_BtoA(real_A)
                    loss_identity_A = self.criterion_identity(identity_A, real_A)
                    loss_identity = (loss_identity_A + loss_identity_B) * lambda_identity
                
                # GAN loss
                fake_B = self.gen_AtoB(real_A)
                pred_fake_B = self.disc_B(fake_B)
                loss_gan_AtoB = self.criterion_gan(pred_fake_B, valid)
                
                fake_A = self.gen_BtoA(real_B)
                pred_fake_A = self.disc_A(fake_A)
                loss_gan_BtoA = self.criterion_gan(pred_fake_A, valid)
                
                loss_gan = (loss_gan_AtoB + loss_gan_BtoA) / 2
                
                # Cycle consistency loss
                recovered_A = self.gen_BtoA(fake_B)
                loss_cycle_A = self.criterion_cycle(recovered_A, real_A)
                
                recovered_B = self.gen_AtoB(fake_A)
                loss_cycle_B = self.criterion_cycle(recovered_B, real_B)
                
                lambda_cycle = self.config['training']['lambda_cycle']
                loss_cycle = (loss_cycle_A + loss_cycle_B) * lambda_cycle
                
                # Total generator loss
                loss_G = loss_gan + loss_cycle + loss_identity
            
            # Backward pass for generators
            if self.use_amp:
                self.scaler.scale(loss_G).backward()
                self.scaler.step(self.optimizer_G)
            else:
                loss_G.backward()
                self.optimizer_G.step()
            
            # ================== Train Discriminators ==================
            self.optimizer_D.zero_grad()
            
            with autocast() if self.use_amp else torch.cuda.amp.autocast(enabled=False):
                # Discriminator A
                pred_real_A = self.disc_A(real_A)
                loss_real_A = self.criterion_gan(pred_real_A, valid)
                
                pred_fake_A = self.disc_A(fake_A.detach())
                loss_fake_A = self.criterion_gan(pred_fake_A, fake)
                
                loss_D_A = (loss_real_A + loss_fake_A) / 2
                
                # Discriminator B
                pred_real_B = self.disc_B(real_B)
                loss_real_B = self.criterion_gan(pred_real_B, valid)
                
                pred_fake_B = self.disc_B(fake_B.detach())
                loss_fake_B = self.criterion_gan(pred_fake_B, fake)
                
                loss_D_B = (loss_real_B + loss_fake_B) / 2
                
                loss_D = loss_D_A + loss_D_B
            
            # Backward pass for discriminators
            if self.use_amp:
                self.scaler.scale(loss_D).backward()
                self.scaler.step(self.optimizer_D)
                self.scaler.update()
            else:
                loss_D.backward()
                self.optimizer_D.step()
            
            # Update metrics
            epoch_losses['G_total'] += loss_G.item()
            epoch_losses['G_gan'] += loss_gan.item()
            epoch_losses['G_cycle'] += loss_cycle.item()
            epoch_losses['G_identity'] += loss_identity if isinstance(loss_identity, float) else loss_identity.item()
            epoch_losses['D_A'] += loss_D_A.item()
            epoch_losses['D_B'] += loss_D_B.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{loss_G.item():.4f}",
                'D': f"{loss_D.item():.4f}"
            })
            
            self.global_step += 1
        
        # Average losses
        n_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Sauvegarde les checkpoints"""
        checkpoint = {
            'epoch': self.current_epoch,
            'gen_AtoB': self.gen_AtoB.state_dict(),
            'gen_BtoA': self.gen_BtoA.state_dict(),
            'disc_A': self.disc_A.state_dict(),
            'disc_B': self.disc_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'best_loss': self.best_loss,
        }
        
        # Sauvegarder checkpoint courant
        torch.save(checkpoint, self.save_dir / 'last.pt')
        
        # Sauvegarder meilleur checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            print(f"✓ Best model saved at epoch {self.current_epoch}")
    
    def train(self):
        """Boucle d'entraînement principale"""
        # Créer dataloader
        train_loader = get_dataloader(
            batch_size=self.config['training']['batch_size'],
            image_size=self.config['training']['image_size'],
            num_workers=self.config['hardware']['num_workers'],
            mode='train',
            use_huggingface=self.config['dataset']['use_huggingface'],
            hf_dataset_name=self.config['dataset']['name']
        )
        
        print(f"\nStarting training for {self.config['training']['epochs']} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Mixed precision: {self.use_amp}\n")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            losses = self.train_epoch(train_loader)
            
            # Log
            print(f"\nEpoch {epoch} Summary:")
            print(f"  G_total: {losses['G_total']:.4f}")
            print(f"  G_gan: {losses['G_gan']:.4f}")
            print(f"  G_cycle: {losses['G_cycle']:.4f}")
            print(f"  D_A: {losses['D_A']:.4f}")
            print(f"  D_B: {losses['D_B']:.4f}")
            
            # Save checkpoint
            is_best = losses['G_total'] < self.best_loss
            if is_best:
                self.best_loss = losses['G_total']
            
            if (epoch + 1) % self.config['checkpointing']['save_every'] == 0:
                self.save_checkpoint(is_best=is_best)
        
        print("\n✅ Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train CycleGAN for Night→Day")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    trainer = CycleGANTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
