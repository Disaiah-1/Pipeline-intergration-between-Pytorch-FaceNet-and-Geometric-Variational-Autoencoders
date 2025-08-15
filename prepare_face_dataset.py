#!/usr/bin/env python3
"""
Face Dataset Preparation Utility
Helps prepare face datasets for the Facenet-RHVAE pipeline
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import argparse
from facenet_rhvae_pipeline import FacenetRHVAEPipeline

def prepare_face_dataset(input_path: str, output_path: str, min_faces: int = 10):
    """
    Prepare a face dataset by detecting and extracting faces
    
    Args:
        input_path: Path to directory containing raw images
        output_path: Path to save processed face images
        min_faces: Minimum number of faces required
    """
    print(f"Preparing face dataset from {input_path} to {output_path}")
    
    # Initialize pipeline for face detection
    pipeline = FacenetRHVAEPipeline()
    pipeline.initialize_facenet()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_path).rglob(f'*{ext}'))
        image_files.extend(Path(input_path).rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} image files")
    
    processed_count = 0
    
    for i, img_path in enumerate(image_files):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Detect and align face
            face_tensor = pipeline.mtcnn(img)
            
            if face_tensor is not None:
                # Convert tensor to PIL image
                face_img = Image.fromarray(
                    (face_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
                )
                
                # Save face image
                output_file = os.path.join(output_path, f"face_{processed_count:04d}.jpg")
                face_img.save(output_file, 'JPEG', quality=95)
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} faces...")
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\n✅ Dataset preparation completed!")
    print(f"Successfully extracted {processed_count} faces")
    print(f"Faces saved to: {output_path}")
    
    if processed_count < min_faces:
        print(f"⚠️  Warning: Only {processed_count} faces extracted (minimum recommended: {min_faces})")
        print("Consider using a larger input dataset for better RHVAE training.")
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description='Prepare face dataset for Facenet-RHVAE pipeline')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to directory containing raw images')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save processed face images')
    parser.add_argument('--min_faces', type=int, default=10,
                       help='Minimum number of faces required')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input path does not exist: {args.input_path}")
        return
    
    prepare_face_dataset(args.input_path, args.output_path, args.min_faces)

if __name__ == "__main__":
    main()
