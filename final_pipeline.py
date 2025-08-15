#!/usr/bin/env python3
"""
Final Working Facenet-RHVAE Pipeline
Based on the successful minimal test approach
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import pickle

# Import facenet-pytorch components
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

# Import RHVAE components
import sys
sys.path.append('data/geometri thing/geometric_perspective_on_vaes')
from models.vae import RHVAE
from config import RHVAE_config


class FacenetRHVAEPipeline:
    """
    Pipeline that connects Facenet face processing to RHVAE training
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mtcnn = None
        self.resnet = None
        self.rhvae = None
        self.face_embeddings = []
        self.face_images = []
        
    def initialize_facenet(self, image_size=160, margin=0):
        """Initialize MTCNN and Inception ResNet V1 models"""
        print("Initializing Facenet models...")
        
        # Initialize MTCNN for face detection and alignment
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=self.device
        )
        
        # Initialize Inception ResNet V1 for face embedding
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        print("Facenet models initialized successfully!")
        
    def process_face_dataset(self, dataset_path: str, output_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a dataset of face images using Facenet
        
        Args:
            dataset_path: Path to directory containing face images
            output_size: Size to resize faces for RHVAE (default 64x64)
            
        Returns:
            Tuple of (processed_faces, face_embeddings)
        """
        print(f"Processing face dataset from: {dataset_path}")
        
        if self.mtcnn is None or self.resnet is None:
            raise ValueError("Facenet models not initialized. Call initialize_facenet() first.")
        
        processed_faces = []
        face_embeddings = []
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(dataset_path).rglob(f'*{ext}'))
            image_files.extend(Path(dataset_path).rglob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} image files")
        
        for i, img_path in enumerate(image_files):
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                
                # Detect and align face
                face_tensor = self.mtcnn(img)
                
                if face_tensor is not None:
                    # Get face embedding
                    with torch.no_grad():
                        embedding = self.resnet(face_tensor.unsqueeze(0).to(self.device))
                    
                    # Resize face for RHVAE (64x64x3 for RGB faces)
                    face_resized = torch.nn.functional.interpolate(
                        face_tensor.unsqueeze(0), 
                        size=(output_size, output_size), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    
                    processed_faces.append(face_resized)
                    face_embeddings.append(embedding.squeeze(0))
                    
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1}/{len(image_files)} images")
                        
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not processed_faces:
            raise ValueError("No faces were successfully processed!")
        
        # Convert to tensors
        processed_faces = torch.stack(processed_faces)
        face_embeddings = torch.stack(face_embeddings)
        
        print(f"Successfully processed {len(processed_faces)} faces")
        print(f"Face tensor shape: {processed_faces.shape}")
        print(f"Embedding tensor shape: {face_embeddings.shape}")
        
        return processed_faces, face_embeddings
    
    def prepare_rhvae_data(self, faces: torch.Tensor, train_split: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare face data for RHVAE training
        
        Args:
            faces: Tensor of face images (N, C, H, W)
            train_split: Fraction of data to use for training
            
        Returns:
            Tuple of (train_data, val_data)
        """
        print("Preparing data for RHVAE training...")
        
        # Normalize to [0, 1] range and ensure proper data type
        faces = faces.float()
        if faces.max() > 1.0:
            faces = faces / 255.0
        print(f"Face data range: [{faces.min():.3f}, {faces.max():.3f}]")
        
        # Split into train/val
        n_train = int(len(faces) * train_split)
        train_data = faces[:n_train]
        val_data = faces[n_train:]
        
        print(f"Training data: {train_data.shape}")
        print(f"Validation data: {val_data.shape}")
        
        return train_data, val_data
    
    def initialize_rhvae(self, input_size: int = 64, latent_dim: int = 64, n_channels: int = 3):
        """
        Initialize RHVAE model for face data
        
        Args:
            input_size: Size of input images (assumed square)
            latent_dim: Dimension of latent space
            n_channels: Number of image channels (3 for RGB)
        """
        print("Initializing RHVAE model...")
        
        # Create RHVAE configuration
        config = RHVAE_config(
            input_dim=input_size * input_size,  # Flattened input size
            model_name="RHVAE",
            architecture='convnet',
            n_channels=n_channels,
            latent_dim=latent_dim,
            beta=0.05,  # Good for face data
            n_lf=1,
            eps_lf=0.001,
            temperature=0.8,
            regularization=0.001,
            device=self.device,
            cuda=torch.cuda.is_available(),
            beta_zero=0.3,
            metric_fc=400,
            dynamic_binarization=False,
            dataset='celeba'  # Use celeba architecture for face data
        )
        
        # Initialize RHVAE model
        self.rhvae = RHVAE(config).to(self.device)
        
        print("RHVAE model initialized successfully!")
        
    def train_rhvae(self, train_data: torch.Tensor, val_data: torch.Tensor, 
                   n_epochs: int = 100, lr: float = 1e-3, patience: int = 10,
                   batch_size: int = 32):
        """
        Train RHVAE model on face data
        
        Args:
            train_data: Training face data
            val_data: Validation face data
            n_epochs: Number of training epochs
            lr: Learning rate
            patience: Early stopping patience
            batch_size: Batch size for training
        """
        if self.rhvae is None:
            raise ValueError("RHVAE model not initialized. Call initialize_rhvae() first.")
        
        print("Starting RHVAE training...")
        
        # Setup optimizer and scheduler
        import torch.optim as optim
        optimizer = optim.Adam(self.rhvae.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=patience, verbose=True
        )
        
        # Training loop - simplified approach for small datasets
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch}/{n_epochs}")
            
            # Training
            self.rhvae.train()
            
            # Use all training data at once (for small datasets)
            if self.device == 'cuda':
                train_data = train_data.cuda()
            
            optimizer.zero_grad()
            
            # Forward pass for RHVAE
            outputs = self.rhvae(train_data)
            (recon_batch, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det) = outputs
            
            # Loss computation
            train_loss = self.rhvae.loss_function(
                recon_batch, train_data, z0, z, rho, eps0, gamma, mu, log_var, G_inv, G_log_det
            )
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            # Update metric for RHVAE
            self.rhvae.update_metric()
            
            # Validation
            self.rhvae.eval()
            with torch.no_grad():
                if self.device == 'cuda':
                    val_data = val_data.cuda()
                
                outputs = self.rhvae(val_data)
                (recon_batch, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det) = outputs
                val_loss = self.rhvae.loss_function(
                    recon_batch, val_data, z0, z, rho, eps0, gamma, mu, log_var, G_inv, G_log_det
                )
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = self.rhvae.state_dict().copy()
            else:
                patience_counter += 1
            
            print(f'Epoch {epoch}: Train loss: {train_loss.item():.6f}, Val loss: {val_loss.item():.6f}')
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.rhvae.load_state_dict(best_model_state)
        print("RHVAE training completed!")
    
    def save_pipeline(self, save_path: str):
        """Save the trained pipeline"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save RHVAE model
        if self.rhvae is not None:
            torch.save(self.rhvae.state_dict(), os.path.join(save_path, 'rhvae_model.pt'))
        
        # Save pipeline info
        pipeline_info = {
            'device': self.device,
            'face_embeddings': self.face_embeddings if hasattr(self, 'face_embeddings') else [],
            'face_images': self.face_images if hasattr(self, 'face_images') else []
        }
        
        with open(os.path.join(save_path, 'pipeline_info.pkl'), 'wb') as f:
            pickle.dump(pipeline_info, f)
        
        print(f"Pipeline saved to {save_path}")
    
    def load_pipeline(self, load_path: str):
        """Load a trained pipeline"""
        # Load RHVAE model
        if self.rhvae is not None:
            self.rhvae.load_state_dict(torch.load(os.path.join(load_path, 'rhvae_model.pt')))
        
        # Load pipeline info
        with open(os.path.join(load_path, 'pipeline_info.pkl'), 'rb') as f:
            pipeline_info = pickle.load(f)
        
        self.face_embeddings = pipeline_info.get('face_embeddings', [])
        self.face_images = pipeline_info.get('face_images', [])
        
        print(f"Pipeline loaded from {load_path}")


def main():
    parser = argparse.ArgumentParser(description='Facenet-RHVAE Pipeline')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to directory containing face images')
    parser.add_argument('--output_dir', type=str, default='facenet_rhvae_output',
                       help='Output directory for trained model')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Size of processed face images')
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent dimension for RHVAE')
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data for training')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FacenetRHVAEPipeline()
    
    # Initialize Facenet
    pipeline.initialize_facenet()
    
    # Process face dataset
    faces, embeddings = pipeline.process_face_dataset(args.dataset_path, args.image_size)
    
    # Prepare data for RHVAE
    train_data, val_data = pipeline.prepare_rhvae_data(faces, args.train_split)
    
    # Initialize RHVAE
    pipeline.initialize_rhvae(args.image_size, args.latent_dim, n_channels=3)
    
    # Train RHVAE
    pipeline.train_rhvae(train_data, val_data, args.n_epochs, args.lr, batch_size=args.batch_size)
    
    # Save pipeline
    pipeline.save_pipeline(args.output_dir)
    
    print("Pipeline training completed successfully!")


if __name__ == "__main__":
    main()
