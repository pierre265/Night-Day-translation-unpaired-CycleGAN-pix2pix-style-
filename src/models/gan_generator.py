"""
CycleGAN Generator - ResNet architecture
Utilisé pour la translation Night→Day

Architecture basée sur le paper CycleGAN original:
https://arxiv.org/abs/1703.10593
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Bloc résiduel avec normalisation et activation"""
    
    def __init__(self, channels: int, use_dropout: bool = False):
        super().__init__()
        
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        
        layers.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # Skip connection


class GeneratorResNet(nn.Module):
    """
    Générateur ResNet pour CycleGAN
    
    Architecture:
    - Downsampling (encoder)
    - Residual blocks (transformation)
    - Upsampling (decoder)
    
    Args:
        input_channels: Nombre de canaux en entrée (3 pour RGB)
        output_channels: Nombre de canaux en sortie (3 pour RGB)
        base_filters: Nombre de filtres de base (64 standard, 32 pour lite)
        n_residual_blocks: Nombre de blocs résiduels (6 ou 9)
        use_dropout: Utiliser dropout dans les blocs résiduels
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_filters: int = 64,
        n_residual_blocks: int = 6,
        use_dropout: bool = False
    ):
        super().__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_filters, kernel_size=7, padding=0),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        ]
        
        # Downsampling
        in_features = base_filters
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features, use_dropout=use_dropout)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, output_channels, kernel_size=7, padding=0),
            nn.Tanh(),  # Output in [-1, 1]
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W] in [0, 1]
        
        Returns:
            Output tensor [B, C, H, W] in [0, 1]
        """
        # Normalize to [-1, 1] for Tanh
        x = x * 2.0 - 1.0
        
        # Generate
        out = self.model(x)
        
        # Denormalize to [0, 1]
        out = (out + 1.0) / 2.0
        
        return out


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator
    
    Classifie des patches de l'image comme réels ou faux
    plutôt que l'image entière
    
    Args:
        input_channels: Nombre de canaux en entrée
        base_filters: Nombre de filtres de base
        n_layers: Nombre de couches (3 standard)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        base_filters: int = 64,
        n_layers: int = 3
    ):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Bloc de convolution pour le discriminateur"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layers = []
        in_filters = input_channels
        
        # First layer (no normalization)
        layers.extend(discriminator_block(in_filters, base_filters, normalize=False))
        in_filters = base_filters
        
        # Intermediate layers
        for i in range(n_layers - 1):
            out_filters = min(in_filters * 2, 512)
            layers.extend(discriminator_block(in_filters, out_filters))
            in_filters = out_filters
        
        # Last layer
        out_filters = min(in_filters * 2, 512)
        layers.extend([
            nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(out_filters),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        # Output layer (PatchGAN output)
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input image [B, C, H, W]
        
        Returns:
            Patch predictions [B, 1, H', W']
        """
        return self.model(x)


def create_generator(
    model_type: str = "resnet_6blocks",
    input_channels: int = 3,
    output_channels: int = 3,
    base_filters: int = 64,
    use_dropout: bool = False
) -> GeneratorResNet:
    """
    Factory function pour créer un générateur
    
    Args:
        model_type: "resnet_6blocks" ou "resnet_9blocks"
        input_channels: Canaux d'entrée
        output_channels: Canaux de sortie
        base_filters: Filtres de base
        use_dropout: Utiliser dropout
    
    Returns:
        Générateur configuré
    """
    n_blocks = 6 if "6blocks" in model_type else 9
    
    return GeneratorResNet(
        input_channels=input_channels,
        output_channels=output_channels,
        base_filters=base_filters,
        n_residual_blocks=n_blocks,
        use_dropout=use_dropout
    )


def create_discriminator(
    input_channels: int = 3,
    base_filters: int = 64,
    n_layers: int = 3
) -> Discriminator:
    """
    Factory function pour créer un discriminateur
    
    Args:
        input_channels: Canaux d'entrée
        base_filters: Filtres de base
        n_layers: Nombre de couches
    
    Returns:
        Discriminateur configuré
    """
    return Discriminator(
        input_channels=input_channels,
        base_filters=base_filters,
        n_layers=n_layers
    )


# Test du modèle
if __name__ == "__main__":
    print("Testing CycleGAN models...")
    
    # Test Generator
    gen = create_generator(model_type="resnet_6blocks")
    x = torch.randn(2, 3, 256, 256)  # Batch de 2 images 256x256
    out = gen(x)
    
    print(f"Generator input shape: {x.shape}")
    print(f"Generator output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Compter les paramètres
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"Generator parameters: {n_params:,} (~{n_params/1e6:.2f}M)")
    
    # Test Discriminator
    disc = create_discriminator()
    disc_out = disc(x)
    print(f"\nDiscriminator output shape: {disc_out.shape}")
    
    n_params_disc = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator parameters: {n_params_disc:,} (~{n_params_disc/1e6:.2f}M)")
    
    print("\n✅ Model test successful!")
