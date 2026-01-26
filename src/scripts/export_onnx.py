"""
Export CycleGAN Generator to ONNX format
Pour optimisation et déploiement sur Jetson avec TensorRT

Usage:
    python src/scripts/export_onnx.py --weights src/runs/night2day_cyclegan/best.pt --out model.onnx
"""

import os
import argparse
import yaml
import torch
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.gan_generator import create_generator


def export_to_onnx(
    config_path: str,
    weights_path: str,
    output_path: str,
    input_size: int = 256,
    opset_version: int = 11,
    simplify: bool = True
):
    """
    Exporte le générateur CycleGAN vers ONNX
    
    Args:
        config_path: Chemin vers config YAML
        weights_path: Chemin vers checkpoint PyTorch
        output_path: Chemin de sortie pour ONNX
        input_size: Taille d'entrée (256, 320, etc.)
        opset_version: Version ONNX opset
        simplify: Simplifier le graph ONNX
    """
    # Charger config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')  # Export sur CPU pour compatibilité
    
    # Créer le générateur
    cfg_gen = config['model']['generator']
    generator = create_generator(
        model_type=cfg_gen['name'],
        input_channels=cfg_gen['input_channels'],
        output_channels=cfg_gen['output_channels'],
        base_filters=cfg_gen['base_filters'],
        use_dropout=False
    ).to(device)
    
    # Charger checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    direction = config['inference']['direction']
    key = 'gen_AtoB' if direction == 'AtoB' else 'gen_BtoA'
    
    generator.load_state_dict(checkpoint[key])
    generator.eval()
    
    print(f"Model loaded: {direction}")
    
    # Dummy input pour tracer le graph
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    
    # Export ONNX
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    torch.onnx.export(
        generator,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        } if config['export'].get('dynamic_axes', False) else None
    )
    
    print(f"✓ ONNX model saved to: {output_path}")
    
    # Simplify ONNX (optionnel, nécessite onnx-simplifier)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified_model, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, output_path)
                print("✓ ONNX model simplified")
            else:
                print("⚠ Simplification check failed, keeping original")
        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")
    
    # Vérifier le modèle ONNX
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
    except ImportError:
        print("⚠ onnx not installed, skipping validation")
    
    # Afficher infos
    print(f"\nModel info:")
    print(f"  Input size: {input_size}x{input_size}")
    print(f"  Input name: 'input'")
    print(f"  Output name: 'output'")
    print(f"  Opset version: {opset_version}")
    
    # Instructions pour TensorRT (optionnel)
    print(f"\nTo convert to TensorRT engine on Jetson:")
    print(f"  trtexec --onnx={output_path} --saveEngine=model.engine --fp16")


def main():
    parser = argparse.ArgumentParser(description="Export CycleGAN to ONNX")
    parser.add_argument('--weights', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='src/configs/night2day.yaml',
                        help='Path to config YAML')
    parser.add_argument('--out', type=str, required=True, help='Output ONNX path')
    parser.add_argument('--size', type=int, default=256, help='Input size')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--no-simplify', action='store_true', help='Disable simplification')
    args = parser.parse_args()
    
    export_to_onnx(
        config_path=args.config,
        weights_path=args.weights,
        output_path=args.out,
        input_size=args.size,
        opset_version=args.opset,
        simplify=not args.no_simplify
    )


if __name__ == "__main__":
    main()
