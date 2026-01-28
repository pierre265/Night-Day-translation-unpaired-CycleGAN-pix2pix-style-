# Supprimer l'ancien fichier et en créer un nouveau
!rm src/datasets/unpaired.py

# Créer le nouveau fichier corrigé
code_corrige = '''"""
Dataloader pour les datasets non-appariés (unpaired) - Projet Night2Day
"""

import os
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset


class UnpairedDataset(Dataset):
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
            print(f"Loading dataset from Hugging Face: {hf_dataset_name}")
            dataset = load_dataset(hf_dataset_name)
            
            # Le dataset a imageA (night) et imageB (day)
            all_data = dataset['train']
            
            if mode == 'train':
                # 90% pour train
                n_train = int(len(all_data) * 0.9)
                self.data = all_data.select(range(n_train))
            else:
                # 10% pour test
                n_train = int(len(all_data) * 0.9)
                self.data = all_data.select(range(n_train, len(all_data)))
            
            self.length = len(self.data)
        else:
            if not domain_a_path or not domain_b_path:
                raise ValueError("Paths required when use_huggingface=False")
            
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
            self.data = None
        
        print(f"Loaded {self.length} images for {mode} mode")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == 'train':
            idx_b = torch.randint(0, self.length, (1,)).item()
        else:
            idx_b = idx
        
        if self.use_huggingface:
            # Accéder à imageA et imageB
            img_a = self.data[idx]['imageA']
            img_b = self.data[idx_b]['imageB']
        else:
            img_a = Image.open(self.images_a[idx]).convert('RGB')
            img_b = Image.open(self.images_b[idx_b]).convert('RGB')
        
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return img_a, img_b


def get_transforms(image_size: int = 256, mode: str = 'train') -> transforms.Compose:
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.12)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
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
        drop_last=(mode == 'train')
    )
    
    return dataloader


if __name__ == "__main__":
    print("Testing UnpairedDataset...")
    
    train_loader = get_dataloader(
        batch_size=4,
        image_size=256,
        num_workers=0,
        mode='train',
        use_huggingface=True
    )
    
    print(f"\\nTrain loader: {len(train_loader)} batches")
    
    img_a, img_b = next(iter(train_loader))
    print(f"Night images: {img_a.shape}")
    print(f"Day images: {img_b.shape}")
    print(f"Range A: [{img_a.min():.3f}, {img_a.max():.3f}]")
    print(f"Range B: [{img_b.min():.3f}, {img_b.max():.3f}]")
    
    print("\\n✅ Dataloader test successful!")
'''

# Écrire le fichier
with open('src/datasets/unpaired.py', 'w') as f:
    f.write(code_corrige)

print("✅ Fichier unpaired.py recréé !")