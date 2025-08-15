import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import math


class InvariantTransformPipeline(nn.Module):
    """
    A pipeline of invariant transformations that can be applied to images
    to make face recognition more robust to variations.
    """
    
    def __init__(self, 
                 enable_rotation=True,
                 enable_scale=True, 
                 enable_brightness=True,
                 enable_contrast=True,
                 enable_noise=True,
                 enable_blur=True,
                 enable_perspective=True,
                 rotation_range=(-15, 15),
                 scale_range=(0.8, 1.2),
                 brightness_range=(0.7, 1.3),
                 contrast_range=(0.7, 1.3),
                 noise_std=0.05,
                 blur_kernel_size=3,
                 perspective_strength=0.1,
                 probability=0.5):
        """
        Initialize the invariant transformation pipeline.
        
        Args:
            enable_rotation (bool): Enable random rotation
            enable_scale (bool): Enable random scaling
            enable_brightness (bool): Enable brightness adjustment
            enable_contrast (bool): Enable contrast adjustment
            enable_noise (bool): Enable Gaussian noise
            enable_blur (bool): Enable Gaussian blur
            enable_perspective (bool): Enable perspective transformation
            rotation_range (tuple): Range of rotation angles in degrees
            scale_range (tuple): Range of scaling factors
            brightness_range (tuple): Range of brightness multipliers
            contrast_range (tuple): Range of contrast multipliers
            noise_std (float): Standard deviation of Gaussian noise
            blur_kernel_size (int): Size of Gaussian blur kernel
            perspective_strength (float): Strength of perspective transformation
            probability (float): Probability of applying each transformation
        """
        super().__init__()
        
        self.enable_rotation = enable_rotation
        self.enable_scale = enable_scale
        self.enable_brightness = enable_brightness
        self.enable_contrast = enable_contrast
        self.enable_noise = enable_noise
        self.enable_blur = enable_blur
        self.enable_perspective = enable_perspective
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_kernel_size = blur_kernel_size
        self.perspective_strength = perspective_strength
        self.probability = probability
        
        # Initialize blur kernel
        if self.enable_blur:
            self.register_buffer('blur_kernel', self._create_gaussian_kernel(blur_kernel_size))
    
    def _create_gaussian_kernel(self, kernel_size):
        """Create a Gaussian blur kernel."""
        sigma = kernel_size / 3.0
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))
        
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _random_rotation(self, x):
        """Apply random rotation to the image."""
        if not self.enable_rotation or random.random() > self.probability:
            return x
        
        angle = random.uniform(*self.rotation_range)
        return self._rotate_tensor(x, angle)
    
    def _rotate_tensor(self, x, angle):
        """Rotate tensor by angle degrees."""
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Create rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # For 2D rotation around center
        h, w = x.shape[-2:]
        center_h, center_w = h / 2, w / 2
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=x.device),
            torch.arange(w, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        
        # Apply rotation transformation
        grid_x = grid_x - center_w
        grid_y = grid_y - center_h
        
        new_x = grid_x * cos_a - grid_y * sin_a + center_w
        new_y = grid_x * sin_a + grid_y * cos_a + center_h
        
        # Normalize to [-1, 1] for grid_sample
        new_x = 2 * new_x / (w - 1) - 1
        new_y = 2 * new_y / (h - 1) - 1
        
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
        
        # Apply grid_sample
        x_rotated = F.grid_sample(x.unsqueeze(0), grid, mode='bilinear', 
                                padding_mode='zeros', align_corners=True)
        return x_rotated.squeeze(0)
    
    def _random_scale(self, x):
        """Apply random scaling to the image."""
        if not self.enable_scale or random.random() > self.probability:
            return x
        
        scale = random.uniform(*self.scale_range)
        h, w = x.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        x_scaled = F.interpolate(x.unsqueeze(0), size=(new_h, new_w), 
                               mode='bilinear', align_corners=False)
        
        # Pad or crop to original size
        if scale > 1:
            # Crop from center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            x_scaled = x_scaled[:, :, start_h:start_h+h, start_w:start_w+w]
        else:
            # Pad with zeros
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            x_scaled = F.pad(x_scaled, (pad_w, w-new_w-pad_w, pad_h, h-new_h-pad_h))
        
        return x_scaled.squeeze(0)
    
    def _random_brightness_contrast(self, x):
        """Apply random brightness and contrast adjustments."""
        if not (self.enable_brightness or self.enable_contrast):
            return x
        
        if self.enable_brightness and random.random() < self.probability:
            brightness = random.uniform(*self.brightness_range)
            x = x * brightness
        
        if self.enable_contrast and random.random() < self.probability:
            contrast = random.uniform(*self.contrast_range)
            mean = x.mean()
            x = (x - mean) * contrast + mean
        
        return x
    
    def _random_noise(self, x):
        """Add random Gaussian noise to the image."""
        if not self.enable_noise or random.random() > self.probability:
            return x
        
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def _random_blur(self, x):
        """Apply random Gaussian blur to the image."""
        if not self.enable_blur or random.random() > self.probability:
            return x
        
        # Apply blur using convolution
        x_blurred = F.conv2d(x.unsqueeze(0), self.blur_kernel.repeat(x.shape[0], 1, 1, 1), 
                           padding=self.blur_kernel_size//2, groups=x.shape[0])
        return x_blurred.squeeze(0)
    
    def _random_perspective(self, x):
        """Apply random perspective transformation to the image."""
        if not self.enable_perspective or random.random() > self.probability:
            return x
        
        h, w = x.shape[-2:]
        
        # Generate random perspective points
        strength = self.perspective_strength
        src_points = torch.tensor([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=torch.float32, device=x.device)
        
        dst_points = src_points + torch.randn(4, 2, device=x.device) * strength * min(h, w)
        
        # Ensure points stay within reasonable bounds
        dst_points[:, 0] = torch.clamp(dst_points[:, 0], 0, w)
        dst_points[:, 1] = torch.clamp(dst_points[:, 1], 0, h)
        
        # Apply perspective transformation using grid_sample
        x_perspective = self._apply_perspective_transform(x, src_points, dst_points)
        return x_perspective
    
    def _apply_perspective_transform(self, x, src_points, dst_points):
        """Apply perspective transform using grid_sample."""
        h, w = x.shape[-2:]
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=x.device),
            torch.arange(w, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        
        # Normalize to [0, 1]
        grid_x = grid_x / (w - 1)
        grid_y = grid_y / (h - 1)
        
        # Apply perspective transformation
        # This is a simplified version - for more accurate results, 
        # we would need to compute the full perspective matrix
        # For now, we'll use a simple bilinear interpolation approach
        
        # Normalize to [-1, 1] for grid_sample
        grid_x = 2 * grid_x - 1
        grid_y = 2 * grid_y - 1
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        # Apply grid_sample
        x_perspective = F.grid_sample(x.unsqueeze(0), grid, mode='bilinear', 
                                    padding_mode='zeros', align_corners=True)
        return x_perspective.squeeze(0)
    
    def forward(self, x):
        """
        Apply invariant transformations to the input tensor.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            torch.Tensor: Transformed image tensor
        """
        # Handle batch dimension
        if len(x.shape) == 4:
            batch_size = x.shape[0]
            results = []
            for i in range(batch_size):
                transformed = self._apply_transforms(x[i])
                results.append(transformed)
            return torch.stack(results)
        else:
            return self._apply_transforms(x)
    
    def _apply_transforms(self, x):
        """Apply all enabled transformations to a single image."""
        # Apply transformations in a specific order
        x = self._random_rotation(x)
        x = self._random_scale(x)
        x = self._random_perspective(x)
        x = self._random_brightness_contrast(x)
        x = self._random_blur(x)
        x = self._random_noise(x)
        
        return x


class AugmentedMTCNN(nn.Module):
    """
    MTCNN with integrated invariant transformations.
    """
    
    def __init__(self, 
                 image_size=160,
                 margin=0,
                 min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7],
                 factor=0.709,
                 post_process=True,
                 keep_all=False,
                 device=None,
                 transform_config=None):
        """
        Initialize AugmentedMTCNN.
        
        Args:
            transform_config (dict): Configuration for invariant transformations
        """
        super().__init__()
        
        # Import MTCNN here to avoid circular imports
        from .mtcnn import MTCNN
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            post_process=post_process,
            keep_all=keep_all,
            device=device
        )
        
        # Initialize transformation pipeline
        if transform_config is None:
            transform_config = {}
        
        self.transform_pipeline = InvariantTransformPipeline(**transform_config)
        
        # Move to device if specified
        if device is not None:
            self.transform_pipeline = self.transform_pipeline.to(device)
    
    def forward(self, img, save_path=None, return_prob=False):
        """
        Detect faces and apply invariant transformations.
        
        Args:
            img: Input image(s)
            save_path: Optional path to save detected faces
            return_prob: Whether to return detection probabilities
            
        Returns:
            Transformed face tensors
        """
        # First, detect and extract faces using MTCNN
        faces = self.mtcnn(img, save_path)
        
        if faces is None:
            return None
        
        # Apply invariant transformations
        if isinstance(faces, list):
            transformed_faces = []
            for face in faces:
                if face is not None:
                    transformed_face = self.transform_pipeline(face)
                    transformed_faces.append(transformed_face)
                else:
                    transformed_faces.append(None)
            return transformed_faces
        else:
            return self.transform_pipeline(faces)
    
    def detect(self, img, landmarks=False):
        """Detect faces without applying transformations."""
        return self.mtcnn.detect(img, landmarks)
    
    def extract(self, img, batch_boxes, save_path):
        """Extract faces without applying transformations."""
        return self.mtcnn.extract(img, batch_boxes, save_path)


class InvariantFaceRecognitionPipeline(nn.Module):
    """
    Complete face recognition pipeline with invariant transformations.
    """
    
    def __init__(self, 
                 pretrained='vggface2',
                 classify=False,
                 num_classes=None,
                 dropout_prob=0.6,
                 device=None,
                 transform_config=None):
        """
        Initialize the complete pipeline.
        
        Args:
            pretrained (str): Pretrained model to use ('vggface2' or 'casia-webface')
            classify (bool): Whether to use classification mode
            num_classes (int): Number of classes for classification
            dropout_prob (float): Dropout probability
            device (torch.device): Device to run on
            transform_config (dict): Configuration for invariant transformations
        """
        super().__init__()
        
        # Import models here to avoid circular imports
        from .inception_resnet_v1 import InceptionResnetV1
        from .mtcnn import MTCNN
        
        # Initialize face detection with transformations
        self.detector = AugmentedMTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            transform_config=transform_config
        )
        
        # Initialize face recognition model
        self.recognizer = InceptionResnetV1(
            pretrained=pretrained,
            classify=classify,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            device=device
        )
        
        self.device = device
    
    def forward(self, img, save_path=None, return_prob=False):
        """
        Complete face recognition pipeline with invariant transformations.
        
        Args:
            img: Input image(s)
            save_path: Optional path to save detected faces
            return_prob: Whether to return detection probabilities
            
        Returns:
            Face embeddings or classification logits
        """
        # Detect faces and apply transformations
        faces = self.detector(img, save_path, return_prob)
        
        if faces is None:
            return None
        
        # Generate embeddings/classifications
        if isinstance(faces, list):
            embeddings = []
            for face in faces:
                if face is not None:
                    # Add batch dimension if needed
                    if len(face.shape) == 3:
                        face = face.unsqueeze(0)
                    embedding = self.recognizer(face)
                    embeddings.append(embedding)
                else:
                    embeddings.append(None)
            return embeddings
        else:
            # Add batch dimension if needed
            if len(faces.shape) == 3:
                faces = faces.unsqueeze(0)
            return self.recognizer(faces)
    
    def get_embeddings(self, img, save_path=None):
        """Get face embeddings with invariant transformations."""
        self.recognizer.classify = False
        return self.forward(img, save_path)
    
    def get_classifications(self, img, save_path=None):
        """Get face classifications with invariant transformations."""
        self.recognizer.classify = True
        return self.forward(img, save_path)
