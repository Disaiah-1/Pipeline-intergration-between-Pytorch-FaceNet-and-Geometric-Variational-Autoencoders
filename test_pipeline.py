#!/usr/bin/env python3
"""
Test script for Facenet-RHVAE Pipeline
Tests the pipeline with minimal data to ensure everything works
"""

import torch
import os
from facenet_rhvae_pipeline import FacenetRHVAEPipeline

def test_pipeline():
    print("=== Testing Facenet-RHVAE Pipeline ===\n")
    
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    pipeline = FacenetRHVAEPipeline(device=device)
    
    # Test 1: Initialize Facenet
    print("\n1. Testing Facenet initialization...")
    try:
        pipeline.initialize_facenet()
        print("✅ Facenet initialized successfully")
    except Exception as e:
        print(f"❌ Facenet initialization failed: {e}")
        return False
    
    # Test 2: Test with a single image
    print("\n2. Testing face processing...")
    test_images_path = "data/test_images"
    
    if not os.path.exists(test_images_path):
        print(f"❌ Test images path not found: {test_images_path}")
        return False
    
    try:
        # Process just a few images for testing
        faces, embeddings = pipeline.process_face_dataset(test_images_path, output_size=64)
        print(f"✅ Successfully processed {len(faces)} faces")
        print(f"   Face tensor shape: {faces.shape}")
        print(f"   Embedding tensor shape: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Face processing failed: {e}")
        return False
    
    # Test 3: Test data preparation
    print("\n3. Testing data preparation...")
    try:
        train_data, val_data = pipeline.prepare_rhvae_data(faces, train_split=0.8)
        print(f"✅ Data preparation successful")
        print(f"   Training data: {train_data.shape}")
        print(f"   Validation data: {val_data.shape}")
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False
    
    # Test 4: Test RHVAE initialization
    print("\n4. Testing RHVAE initialization...")
    try:
        pipeline.initialize_rhvae(input_size=64, latent_dim=32, n_channels=3)
        print("✅ RHVAE initialized successfully")
    except Exception as e:
        print(f"❌ RHVAE initialization failed: {e}")
        return False
    
    # Test 5: Test minimal training (1 epoch)
    print("\n5. Testing minimal training...")
    try:
        pipeline.train_rhvae(
            train_data, 
            val_data, 
            n_epochs=1,  # Just 1 epoch for testing
            lr=1e-3, 
            patience=1,
            batch_size=1  # Small batch size for testing
        )
        print("✅ Training completed successfully")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Test 6: Test saving
    print("\n6. Testing model saving...")
    try:
        output_dir = "test_output"
        pipeline.save_pipeline(output_dir)
        print(f"✅ Pipeline saved to {output_dir}")
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Pipeline is working correctly.")
    return True

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        print("\n✅ Pipeline test completed successfully!")
    else:
        print("\n❌ Pipeline test failed!")
