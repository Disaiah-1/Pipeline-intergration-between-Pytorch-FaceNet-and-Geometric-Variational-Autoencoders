# Facenet-RHVAE Pipeline

This pipeline connects **Facenet** face detection and embedding with **RHVAE** (Riemannian Hamiltonian VAE) training, allowing you to train a geometric VAE on face datasets.

## Overview

The pipeline performs the following steps:
1. **Face Detection**: Uses MTCNN to detect and align faces in images
2. **Face Embedding**: Uses Inception ResNet V1 to extract face embeddings
3. **Data Preparation**: Processes faces for RHVAE training (resize, normalize)
4. **RHVAE Training**: Trains a Riemannian Hamiltonian VAE on the face data
5. **Model Saving**: Saves the trained pipeline for later use

## Features

- ✅ **Automatic face detection and alignment** using MTCNN
- ✅ **Face embedding extraction** using Inception ResNet V1
- ✅ **Seamless integration** with existing RHVAE implementation
- ✅ **Configurable parameters** for different use cases
- ✅ **Save/load functionality** for trained models
- ✅ **Batch processing** for large datasets

## Requirements

The pipeline uses the virtual environment we created earlier. Make sure to activate it:

```bash
source venv/bin/activate
```

## Quick Start

### 1. Basic Usage

```python
from facenet_rhvae_pipeline import FacenetRHVAEPipeline

# Initialize pipeline
pipeline = FacenetRHVAEPipeline()

# Initialize Facenet models
pipeline.initialize_facenet()

# Process face dataset
faces, embeddings = pipeline.process_face_dataset("path/to/face/images", output_size=64)

# Prepare data for RHVAE
train_data, val_data = pipeline.prepare_rhvae_data(faces)

# Initialize and train RHVAE
pipeline.initialize_rhvae(input_size=64, latent_dim=64, n_channels=3)
pipeline.train_rhvae(train_data, val_data, n_epochs=100)

# Save pipeline
pipeline.save_pipeline("output_directory")
```

### 2. Command Line Usage

```bash
# Run the full pipeline
python facenet_rhvae_pipeline.py --dataset_path /path/to/face/images --output_dir output

# Run with custom parameters
python facenet_rhvae_pipeline.py \
    --dataset_path /path/to/face/images \
    --output_dir output \
    --image_size 64 \
    --latent_dim 64 \
    --n_epochs 100 \
    --lr 1e-3 \
    --batch_size 32
```

### 3. Example with Test Images

```bash
# Run the example with existing test images
python example_facenet_rhvae.py
```

## Dataset Preparation

### Option 1: Direct Processing
The pipeline can process raw images directly:

```bash
python facenet_rhvae_pipeline.py --dataset_path /path/to/raw/images
```

### Option 2: Pre-process Faces
Extract faces first, then train:

```bash
# Extract faces from raw images
python prepare_face_dataset.py --input_path /path/to/raw/images --output_path /path/to/faces

# Train on extracted faces
python facenet_rhvae_pipeline.py --dataset_path /path/to/faces
```

## Configuration

### RHVAE Parameters

The pipeline automatically configures RHVAE for face data:

- **Input size**: 64x64 (configurable)
- **Latent dimension**: 64 (configurable)
- **Channels**: 3 (RGB)
- **Beta**: 0.05 (good for face data)
- **Temperature**: 0.8
- **Regularization**: 0.001

### Custom Configuration

You can modify the RHVAE configuration in the `initialize_rhvae` method:

```python
# Custom RHVAE configuration
config = RHVAE_config(
    input_dim=64,
    latent_dim=128,  # Larger latent space
    beta=0.1,        # Different beta value
    temperature=1.0, # Different temperature
    # ... other parameters
)
```

## File Structure

```
facenet-pytorch/
├── facenet_rhvae_pipeline.py      # Main pipeline
├── example_facenet_rhvae.py       # Example usage
├── prepare_face_dataset.py        # Dataset preparation utility
├── FACENET_RHVAE_README.md        # This file
└── data/
    └── test_images/               # Test images for demo
```

## Output

The pipeline saves:
- `rhvae_model.pt`: Trained RHVAE model weights
- `pipeline_info.pkl`: Pipeline metadata and embeddings

## Troubleshooting

### Common Issues

1. **No faces detected**: Ensure images contain clear, front-facing faces
2. **Memory issues**: Reduce batch size or image size
3. **CUDA errors**: Check GPU memory or use CPU mode
4. **Import errors**: Ensure virtual environment is activated

### Performance Tips

- Use GPU for faster training (automatically detected)
- Increase batch size if memory allows
- Use larger datasets for better results
- Adjust learning rate based on convergence

## Integration with Existing RHVAE

The pipeline is designed to work with the existing RHVAE implementation without modifications:

- ✅ Uses existing `RHVAE` class
- ✅ Uses existing `train_vae` function
- ✅ Uses existing configuration system
- ✅ Maintains all original RHVAE functionality

## Advanced Usage

### Custom Face Processing

```python
# Custom face processing
def custom_face_processor(image):
    # Your custom processing logic
    return processed_face

# Use in pipeline
pipeline.process_face_dataset_with_custom_processor(
    dataset_path, 
    custom_face_processor
)
```

### Multiple Datasets

```python
# Process multiple datasets
datasets = ["dataset1", "dataset2", "dataset3"]
all_faces = []

for dataset in datasets:
    faces, _ = pipeline.process_face_dataset(dataset)
    all_faces.append(faces)

# Combine datasets
combined_faces = torch.cat(all_faces, dim=0)
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{facenet_rhvae_pipeline,
  title={Facenet-RHVAE Pipeline: Connecting Face Detection to Geometric VAE Training},
  author={Your Name},
  year={2024}
}
```

## License

This pipeline is provided as-is for research and educational purposes.
