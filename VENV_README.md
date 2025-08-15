# Virtual Environment Setup

This virtual environment has been configured to satisfy the requirements for both:
1. **facenet-pytorch** - Face detection and recognition models
2. **geometric thing** - Geometric perspective on VAEs

## Setup

The virtual environment is already created and configured. To activate it:

```bash
# Option 1: Use the activation script
./activate_env.sh

# Option 2: Activate manually
source venv/bin/activate
```

## Installed Packages

### Core Dependencies (from facenet-pytorch)
- `numpy>=1.24.0,<2.0.0`
- `Pillow>=10.2.0,<10.3.0`
- `requests>=2.0.0,<3.0.0`
- `torch>=2.2.0,<=2.3.0` (PyTorch 2.2.2)
- `torchvision>=0.17.0,<=0.18.0` (torchvision 0.17.2)
- `tqdm>=4.0.0,<5.0.0`

### Additional Dependencies (from geometric thing)
- `matplotlib>=3.3.4`
- `h5py>=2.10.0`
- `imageio>=2.8.0`
- `scipy`
- `scikit_learn_extra>=0.2.0`
- `scikit_learn`

## Version Compatibility

**Note**: The geometric thing originally required older PyTorch versions (torch==1.9.0, torchvision==0.10.0), but this environment uses newer versions (torch>=2.2.0, torchvision>=0.17.0) to satisfy facenet-pytorch requirements. PyTorch maintains backward compatibility, so the geometric thing should work with these newer versions.

## Usage

After activating the virtual environment:

1. **For facenet-pytorch**: The package is installed in development mode, so you can import it directly:
   ```python
   import facenet_pytorch
   ```

2. **For geometric thing**: Navigate to the geometric thing directory and run the scripts:
   ```bash
   cd "data/geometri thing/geometric_perspective_on_vaes"
   python train_vae.py  # or other scripts
   ```

## Deactivation

To deactivate the virtual environment:
```bash
deactivate
```

## Files Created

- `venv/` - The virtual environment directory
- `combined_requirements.txt` - Combined requirements file
- `activate_env.sh` - Activation script
- `VENV_README.md` - This documentation file
