"""
Dataloader pour les datasets non-appariés (unpaired) - Projet Night2Day
Utilisé pour CycleGAN et autres approches de domain translation

Ce dataloader charge des images de deux domaines différents (A et B)
sans correspondance explicite entre elles.
"""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset


class UnpairedDataset(Dataset):
    """
    Dataset pour chargement d'images non-appariées (deux domaines séparés)
    
    Args:
        domain_a_path: Chemin vers les images du domaine A (ex: night)
        domain_b_path: Chemin vers les images du domaine B (ex: day)
        transform: Transformations à appliquer
        mode: 'train' ou 'test'
    """
    
    def __init__(
        self,
        domain_a_path: Optional[str] = None,
        domain_b_path: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        mode: str = 'train',
        use_huggingface: bool = True,
        hf_dataset_name: str = "huggan/night2day"
    ):
        self.mode = mode
        self.transform = transform
        self.use_huggingface = use_huggingface
        
        if use_huggingface:
            # Chargement depuis Hugging Face
            print(f"Loading dataset from Hugging Face: {hf_dataset_name}")
            dataset = load_dataset(hf_dataset_name)
            
            # Extraction des images selon le mode
            if mode == 'train':
                self.images_a = dataset['train']  # Night images
                self.images_b = dataset['train']  # Day images (même split mais domaine différent)
            else:
                self.images_a = dataset['test'] if 'test' in dataset else dataset['train']
                self.images_b = dataset['test'] if 'test' in dataset else dataset['train']
                
            self.length = min(len(self.images_a), len(self.images_b))
            
        else:
            # Chargement depuis dossiers locaux (fallback)
            if not domain_a_path or not domain_b_path:
                raise ValueError("domain_a_path and domain_b_path required when use_huggingface=False")
            
            self.images_a = sorted([
                os.path.join(domain_a_path, f) 
                for f in os.listdir(domain_a_path) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.images_b = sorted([
                os.path.join(domain_b_path, f) 
                for f in os.listdir(domain_b_path) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.length = min(len(self.images_a), len(self.images_b))
        
        print(f"Loaded {self.length} images per domain for {mode} mode")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne un tuple (image_a, image_b)
        Les images ne sont PAS appariées (domaines différents)
        """
        # Pour unpaired data, on peut randomiser l'index du domaine B
        if self.mode == 'train':
            idx_b = torch.randint(0, self.length, (1,)).item()
        else:
            idx_b = idx  # Fixe pour test (reproductibilité)
        
        # Chargement des images
        if self.use_huggingface:
            img_a = self.images_a[idx]['image']  # PIL Image
            img_b = self.images_b[idx_b]['image']  # PIL Image
        else:
            img_a = Image.open(self.images_a[idx]).convert('RGB')
            img_b = Image.open(self.images_b[idx_b]).convert('RGB')
        
        # Application des transformations
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return img_a, img_b


def get_transforms(image_size: int = 256, mode: str = 'train') -> transforms.Compose:
    """
    Retourne les transformations appropriées selon le mode
    
    Args:
        image_size: Taille cible des images
        mode: 'train' ou 'test'
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.12)),  # Slightly larger for crop
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # [0, 1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])


def get_dataloader(
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 4,
    mode: str = 'train',
    use_huggingface: bool = True,
    domain_a_path: Optional[str] = None,
    domain_b_path: Optional[str] = None,
    hf_dataset_name: str = "huggan/night2day"
) -> DataLoader:
    """
    Crée un DataLoader pour le dataset unpaired
    
    Args:
        batch_size: Taille des batchs
        image_size: Taille des images après resize
        num_workers: Nombre de workers pour le chargement
        mode: 'train' ou 'test'
        use_huggingface: Utiliser le dataset HuggingFace ou local
        domain_a_path: Chemin local domaine A (si use_huggingface=False)
        domain_b_path: Chemin local domaine B (si use_huggingface=False)
        hf_dataset_name: Nom du dataset HuggingFace
    
    Returns:
        DataLoader configuré
    """
    transform = get_transforms(image_size, mode)
    
    dataset = UnpairedDataset(
        domain_a_path=domain_a_path,
        domain_b_path=domain_b_path,
        transform=transform,
        mode=mode,
        use_huggingface=use_huggingface,
        hf_dataset_name=hf_dataset_name
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train')  # Drop last incomplete batch in training
    )
    
    return dataloader


# Test du dataloader
if __name__ == "__main__":
    print("Testing UnpairedDataset...")
    
    # Test avec HuggingFace
    train_loader = get_dataloader(
        batch_size=4,
        image_size=256,
        num_workers=0,  # 0 pour debugging
        mode='train',
        use_huggingface=True
    )
    
    print(f"\nTrain loader created: {len(train_loader)} batches")
    
    # Test d'un batch
    img_a, img_b = next(iter(train_loader))
    print(f"Batch shape - Domain A (night): {img_a.shape}")
    print(f"Batch shape - Domain B (day): {img_b.shape}")
    print(f"Value range - Domain A: [{img_a.min():.3f}, {img_a.max():.3f}]")
    print(f"Value range - Domain B: [{img_b.min():.3f}, {img_b.max():.3f}]")
    
    print("\n✅ Dataloader test successful!")
