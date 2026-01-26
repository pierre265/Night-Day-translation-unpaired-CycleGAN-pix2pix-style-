"""
Demo live split-screen pour Jetson - Night→Day Translation
Phase 3: Jetson integration with real-time split-screen demo

Usage:
    python src/demo_live_split.py --weights src/runs/night2day_cyclegan/best.pt --size 256

Controls:
    - Press 'q' to quit
    - Press 's' to save current frame
"""

import os
import argparse
import yaml
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.gan_generator import create_generator


class LiveDemo:
    """Demo live pour CycleGAN sur Jetson"""
    
    def __init__(self, config_path: str, weights_path: str, inference_size: int = 256):
        # Charger config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'])
        self.inference_size = inference_size
        
        # Charger le modèle
        self._load_model(weights_path)
        
        # Initialiser la caméra
        self._init_camera()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((inference_size, inference_size)),
            transforms.ToTensor(),
        ])
        
        # FPS tracking
        self.fps_list = []
        
        print(f"Demo initialized - Device: {self.device}, Size: {inference_size}x{inference_size}")
    
    def _load_model(self, weights_path: str):
        """Charge le générateur"""
        cfg_gen = self.config['model']['generator']
        
        self.generator = create_generator(
            model_type=cfg_gen['name'],
            input_channels=cfg_gen['input_channels'],
            output_channels=cfg_gen['output_channels'],
            base_filters=cfg_gen['base_filters'],
            use_dropout=False
        ).to(self.device)
        
        # Charger checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)
        direction = self.config['inference']['direction']
        key = 'gen_AtoB' if direction == 'AtoB' else 'gen_BtoA'
        
        self.generator.load_state_dict(checkpoint[key])
        self.generator.eval()
        
        print(f"Model loaded: {direction} ({'Night→Day' if direction == 'AtoB' else 'Day→Night'})")
    
    def _init_camera(self):
        """Initialise la caméra (CSI ou USB)"""
        # Essayer CSI camera (Jetson)
        gst_pipeline = (
            "nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={self.config['jetson']['camera_width']}, "
            f"height={self.config['jetson']['camera_height']}, "
            f"framerate={self.config['jetson']['camera_fps']}/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        # Fallback to USB camera
        if not self.cap.isOpened():
            print("CSI camera not available, trying USB camera...")
            self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera!")
        
        print("Camera initialized")
    
    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Traite une frame avec le générateur
        
        Args:
            frame: Frame BGR de OpenCV
        
        Returns:
            Frame traitée (RGB, 0-255)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform et envoyer au device
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Inférence
        start = time.time()
        output_tensor = self.generator(input_tensor)
        inference_time = time.time() - start
        
        # Convert back to numpy
        output_np = output_tensor.squeeze(0).cpu().numpy()
        output_np = np.transpose(output_np, (1, 2, 0))  # CHW -> HWC
        output_np = (output_np * 255).astype(np.uint8)
        
        # Calculer FPS
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_list.append(fps)
        if len(self.fps_list) > 30:
            self.fps_list.pop(0)
        
        return output_np, np.mean(self.fps_list)
    
    def create_split_screen(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        fps: float
    ) -> np.ndarray:
        """
        Crée l'affichage split-screen
        
        Args:
            left_frame: Frame originale (BGR)
            right_frame: Frame traitée (RGB)
            fps: FPS actuel
        
        Returns:
            Frame split-screen avec overlay FPS
        """
        # Resize frames to same size
        h, w = self.inference_size, self.inference_size
        left_resized = cv2.resize(left_frame, (w, h))
        right_resized = cv2.resize(right_frame, (w, h))
        
        # Convert right to BGR for display
        right_bgr = cv2.cvtColor(right_resized, cv2.COLOR_RGB2BGR)
        
        # Create split screen (horizontal concat)
        split = np.concatenate([left_resized, right_bgr], axis=1)
        
        # Add labels
        cv2.putText(split, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(split, "Night→Day", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add FPS
        if self.config['jetson']['display_fps']:
            cv2.putText(split, f"FPS: {fps:.1f}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return split
    
    def run(self):
        """Lance la démo live"""
        print("\n=== Live Demo Started ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame\n")
        
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process frame
                processed, fps = self.process_frame(frame)
                
                # Create split screen display
                display = self.create_split_screen(frame, processed, fps)
                
                # Show
                cv2.imshow('Night→Day Translation Demo', display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"frame_{frame_count:04d}.png"
                    cv2.imwrite(save_path, display)
                    print(f"Saved: {save_path}")
                
                frame_count += 1
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            
            if self.fps_list:
                avg_fps = np.mean(self.fps_list)
                print(f"\nAverage FPS: {avg_fps:.2f}")
            
            print("Demo terminated")


def main():
    parser = argparse.ArgumentParser(description="Live demo for Night→Day translation")
    parser.add_argument('--weights', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='src/configs/night2day.yaml',
                        help='Path to config file')
    parser.add_argument('--size', type=int, default=256,
                        help='Inference size (128, 192, 256, 320)')
    args = parser.parse_args()
    
    # Run demo
    demo = LiveDemo(
        config_path=args.config,
        weights_path=args.weights,
        inference_size=args.size
    )
    demo.run()


if __name__ == "__main__":
    main()
