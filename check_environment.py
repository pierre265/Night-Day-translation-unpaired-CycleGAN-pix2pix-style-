"""
Script de vérification de l'environnement
Vérifie que toutes les dépendances sont installées et fonctionnelles

Usage:
    python check_environment.py
"""

import sys
import importlib.util

def check_package(package_name, import_name=None, min_version=None):
    """Vérifie si un package est installé"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            # Simple version check
            if version < min_version:
                print(f"⚠️  {package_name}: {version} (minimum {min_version} recommandé)")
                return False
        
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False

def check_cuda():
    """Vérifie CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available (version {torch.version.cuda})")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  CUDA: Not available (CPU only)")
            return False
    except:
        print("❌ CUDA: Error checking")
        return False

def check_mps():
    """Vérifie Apple MPS (Metal Performance Shaders)"""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple MPS: Available")
            return True
        else:
            return False
    except:
        return False

def main():
    print("=" * 60)
    print("🔍 Vérification de l'environnement - Night→Day Project")
    print("=" * 60)
    
    print("\n📦 Packages Python:")
    print("-" * 60)
    
    packages = [
        ('torch', None, '2.0.0'),
        ('torchvision', None, '0.15.0'),
        ('numpy', None, '1.23.0'),
        ('PIL', 'PIL', None),
        ('cv2', 'cv2', None),
        ('yaml', 'yaml', None),
        ('datasets', None, '2.14.0'),
        ('huggingface_hub', None, None),
        ('tqdm', None, None),
    ]
    
    all_ok = True
    for pkg in packages:
        if not check_package(*pkg):
            all_ok = False
    
    print("\n🖥️  Hardware:")
    print("-" * 60)
    
    cuda_available = check_cuda()
    mps_available = check_mps()
    
    if not cuda_available and not mps_available:
        print("⚠️  Attention: Ni CUDA ni MPS disponible. Entraînement sur CPU sera TRÈS lent.")
    
    print("\n📁 Structure du projet:")
    print("-" * 60)
    
    import os
    required_dirs = [
        'src/configs',
        'src/datasets',
        'src/models',
        'src/scripts',
        'data'
    ]
    
    required_files = [
        'src/configs/night2day.yaml',
        'src/datasets/unpaired.py',
        'src/models/gan_generator.py',
        'src/train.py',
        'src/eval.py',
        'src/demo_live_split.py',
        'requirements.txt',
        'README.md'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - MANQUANT")
            all_ok = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MANQUANT")
            all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ Environnement prêt! Vous pouvez commencer l'entraînement.")
        print("\nProchaines étapes:")
        print("  1. Tester le dataloader: python src/datasets/unpaired.py")
        print("  2. Tester le modèle: python src/models/gan_generator.py")
        print("  3. Lancer training: python src/train.py --config src/configs/night2day.yaml")
    else:
        print("❌ Certains éléments sont manquants. Vérifiez les messages ci-dessus.")
        print("\nPour installer les dépendances:")
        print("  pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
