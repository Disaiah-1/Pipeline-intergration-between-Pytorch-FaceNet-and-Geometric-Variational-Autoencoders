#!/usr/bin/env python3
"""
Debug test to identify gradient issues
"""

import torch
import sys
sys.path.append('data/geometri thing/geometric_perspective_on_vaes')
from models.vae import RHVAE
from config import RHVAE_config

def debug_rhvae():
    print("Debugging RHVAE gradient issues...")
    
    # Create simple configuration
    config = RHVAE_config(
        input_dim=64 * 64,
        model_name="RHVAE",
        architecture='convnet',
        n_channels=3,
        latent_dim=32,
        beta=0.05,
        n_lf=1,
        eps_lf=0.001,
        temperature=0.8,
        regularization=0.001,
        device='cpu',
        cuda=False,
        beta_zero=0.3,
        metric_fc=400,
        dynamic_binarization=False,
        dataset='celeba'
    )
    
    # Initialize RHVAE
    rhvae = RHVAE(config)
    print("✅ RHVAE initialized")
    
    # Create dummy data with gradients
    batch_size = 2
    data = torch.rand(batch_size, 3, 64, 64, requires_grad=True)
    print(f"✅ Created dummy data: {data.shape}, requires_grad: {data.requires_grad}")
    
    # Test forward pass
    try:
        rhvae.train()
        outputs = rhvae(data)
        print(f"✅ Forward pass successful, outputs: {len(outputs)}")
        
        # Check which outputs require gradients
        for i, output in enumerate(outputs):
            print(f"   Output {i}: shape={output.shape}, requires_grad={output.requires_grad}")
        
        # Test loss function
        (recon_x, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det) = outputs
        loss = rhvae.loss_function(recon_x, data, z0, z, rho, eps0, gamma, mu, log_var, G_inv, G_log_det)
        print(f"✅ Loss computation successful: {loss.item()}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        
        # Test backward pass
        loss.backward()
        print("✅ Backward pass successful")
        
        # Check gradients
        for name, param in rhvae.named_parameters():
            if param.grad is not None:
                print(f"   {name}: grad_norm={param.grad.norm().item()}")
            else:
                print(f"   {name}: no gradient")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rhvae()
