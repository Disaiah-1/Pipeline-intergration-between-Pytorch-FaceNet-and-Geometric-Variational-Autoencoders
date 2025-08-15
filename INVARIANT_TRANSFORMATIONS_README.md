# Invariant Transformations for Face Recognition

This document describes the new invariant transformation system that has been added to the facenet-pytorch library to make face recognition more robust to variations in input images.

## Overview

The invariant transformation system applies various geometric and photometric transformations to images before they are processed by the face recognition pipeline. This helps the system become more robust to:

- **Geometric variations**: Rotation, scaling, perspective changes
- **Photometric variations**: Brightness, contrast, noise, blur
- **Real-world conditions**: Different lighting, camera angles, image quality

## Components

### 1. InvariantTransformPipeline

The core transformation module that applies various transformations to image tensors.

```python
from facenet_pytorch import InvariantTransformPipeline

# Configure transformations
transform_config = {
    'enable_rotation': True,
    'enable_scale': True,
    'enable_brightness': True,
    'enable_contrast': True,
    'enable_noise': True,
    'enable_blur': True,
    'enable_perspective': True,
    'rotation_range': (-15, 15),
    'scale_range': (0.8, 1.2),
    'brightness_range': (0.7, 1.3),
    'contrast_range': (0.7, 1.3),
    'noise_std': 0.05,
    'blur_kernel_size': 3,
    'perspective_strength': 0.1,
    'probability': 0.5
}

# Initialize pipeline
transform_pipeline = InvariantTransformPipeline(**transform_config)

# Apply transformations
transformed_image = transform_pipeline(image_tensor)
```

### 2. AugmentedMTCNN

MTCNN face detector with integrated invariant transformations.

```python
from facenet_pytorch import AugmentedMTCNN

# Initialize with transformations
augmented_mtcnn = AugmentedMTCNN(
    image_size=160,
    margin=0,
    device=device,
    transform_config=transform_config
)

# Detect faces with transformations applied
faces = augmented_mtcnn(image)
```

### 3. InvariantFaceRecognitionPipeline

Complete face recognition pipeline with invariant transformations.

```python
from facenet_pytorch import InvariantFaceRecognitionPipeline

# Initialize complete pipeline
pipeline = InvariantFaceRecognitionPipeline(
    pretrained='vggface2',
    device=device,
    transform_config=transform_config
)

# Get embeddings with transformations
embeddings = pipeline.get_embeddings(image)

# Get classifications with transformations
classifications = pipeline.get_classifications(image)
```

## Available Transformations

### Geometric Transformations

1. **Rotation** (`enable_rotation`)
   - Random rotation within specified range
   - Range: `rotation_range=(-15, 15)` degrees
   - Uses bilinear interpolation

2. **Scaling** (`enable_scale`)
   - Random scaling with crop/pad to maintain size
   - Range: `scale_range=(0.8, 1.2)`
   - Centers the crop/pad operation

3. **Perspective** (`enable_perspective`)
   - Random perspective transformation
   - Strength: `perspective_strength=0.1`
   - Maintains image boundaries

### Photometric Transformations

1. **Brightness** (`enable_brightness`)
   - Multiplicative brightness adjustment
   - Range: `brightness_range=(0.7, 1.3)`

2. **Contrast** (`enable_contrast`)
   - Contrast adjustment around mean
   - Range: `contrast_range=(0.7, 1.3)`

3. **Noise** (`enable_noise`)
   - Additive Gaussian noise
   - Standard deviation: `noise_std=0.05`

4. **Blur** (`enable_blur`)
   - Gaussian blur with configurable kernel
   - Kernel size: `blur_kernel_size=3`

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_rotation` | bool | True | Enable random rotation |
| `enable_scale` | bool | True | Enable random scaling |
| `enable_brightness` | bool | True | Enable brightness adjustment |
| `enable_contrast` | bool | True | Enable contrast adjustment |
| `enable_noise` | bool | True | Enable Gaussian noise |
| `enable_blur` | bool | True | Enable Gaussian blur |
| `enable_perspective` | bool | True | Enable perspective transformation |
| `rotation_range` | tuple | (-15, 15) | Rotation angle range in degrees |
| `scale_range` | tuple | (0.8, 1.2) | Scaling factor range |
| `brightness_range` | tuple | (0.7, 1.3) | Brightness multiplier range |
| `contrast_range` | tuple | (0.7, 1.3) | Contrast multiplier range |
| `noise_std` | float | 0.05 | Standard deviation of Gaussian noise |
| `blur_kernel_size` | int | 3 | Size of Gaussian blur kernel |
| `perspective_strength` | float | 0.1 | Strength of perspective transformation |
| `probability` | float | 0.5 | Probability of applying each transformation |

## Usage Examples

### Basic Usage

```python
import torch
from facenet_pytorch import InvariantFaceRecognitionPipeline
from PIL import Image

# Load image
image = Image.open('face.jpg')

# Configure transformations
transform_config = {
    'enable_rotation': True,
    'enable_scale': True,
    'enable_brightness': True,
    'enable_contrast': True,
    'enable_noise': True,
    'enable_blur': True,
    'enable_perspective': True,
    'rotation_range': (-10, 10),
    'scale_range': (0.9, 1.1),
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2),
    'noise_std': 0.03,
    'blur_kernel_size': 3,
    'perspective_strength': 0.05,
    'probability': 0.6
}

# Initialize pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline = InvariantFaceRecognitionPipeline(
    pretrained='vggface2',
    device=device,
    transform_config=transform_config
)

# Get embeddings
embeddings = pipeline.get_embeddings(image)
print(f"Embedding shape: {embeddings.shape}")
```

### Custom Transformation Pipeline

```python
from facenet_pytorch import InvariantTransformPipeline
from torchvision import transforms

# Create custom transformation pipeline
transform_pipeline = InvariantTransformPipeline(
    enable_rotation=True,
    enable_scale=False,  # Disable scaling
    enable_brightness=True,
    enable_contrast=True,
    enable_noise=False,  # Disable noise
    enable_blur=True,
    enable_perspective=False,  # Disable perspective
    rotation_range=(-5, 5),  # Smaller rotation range
    brightness_range=(0.9, 1.1),  # Smaller brightness range
    contrast_range=(0.9, 1.1),  # Smaller contrast range
    blur_kernel_size=5,  # Larger blur kernel
    probability=0.8  # Higher probability
)

# Apply to image tensor
to_tensor = transforms.ToTensor()
image_tensor = to_tensor(image)

with torch.no_grad():
    transformed = transform_pipeline(image_tensor)
```

### Batch Processing

```python
# Process multiple images
images = [Image.open(f'face_{i}.jpg') for i in range(5)]

# Get embeddings for all images
embeddings_list = []
for image in images:
    embedding = pipeline.get_embeddings(image)
    if embedding is not None:
        embeddings_list.append(embedding)

if embeddings_list:
    embeddings_batch = torch.cat(embeddings_list, dim=0)
    print(f"Batch embeddings shape: {embeddings_batch.shape}")
```

## Integration with Existing Code

The invariant transformation system is designed to be a drop-in replacement for the existing MTCNN and InceptionResnetV1 components. You can easily switch between the standard and invariant versions:

### Standard Pipeline
```python
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

faces = mtcnn(image)
embeddings = resnet(faces)
```

### Invariant Pipeline
```python
from facenet_pytorch import InvariantFaceRecognitionPipeline

pipeline = InvariantFaceRecognitionPipeline(
    pretrained='vggface2',
    device=device,
    transform_config=transform_config
)

embeddings = pipeline.get_embeddings(image)
```

## Performance Considerations

1. **Computational Overhead**: Transformations add computational cost, but this is typically minimal compared to the neural network inference.

2. **Memory Usage**: Transformations are applied in-place when possible to minimize memory usage.

3. **GPU Acceleration**: All transformations are GPU-compatible and will automatically use GPU if available.

4. **Batch Processing**: The system efficiently handles batch processing of multiple images.

## Testing

Run the test script to verify the system works correctly:

```bash
python test_invariant_transforms.py
```

Run the comprehensive example:

```bash
python examples/invariant_face_recognition_example.py
```

## Benefits

1. **Improved Robustness**: Better performance on images with variations in lighting, angle, and quality.

2. **Data Augmentation**: Can be used for training data augmentation.

3. **Real-world Performance**: More reliable in real-world scenarios with imperfect images.

4. **Configurable**: Easy to enable/disable specific transformations based on your needs.

5. **Backward Compatible**: Can be used alongside existing code without breaking changes.

## Limitations

1. **Computational Cost**: Slight increase in processing time due to transformations.

2. **Memory Usage**: Small increase in memory usage for transformation kernels.

3. **Parameter Tuning**: May require tuning of transformation parameters for optimal performance.

## Future Enhancements

Potential future improvements could include:

1. **Adaptive Transformations**: Transformations that adapt based on image content.
2. **Learning-based Augmentation**: Transformations learned from training data.
3. **Domain-specific Transformations**: Transformations tailored for specific use cases.
4. **Real-time Optimization**: Optimizations for real-time applications.

## Support

For issues or questions about the invariant transformation system, please refer to the main facenet-pytorch documentation or create an issue in the repository.
