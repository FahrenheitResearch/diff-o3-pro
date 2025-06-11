#!/usr/bin/env python
"""
Unit tests for forward pass of forecast and diffusion models.
Tests model initialization, forward pass, and output shapes.
"""
import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet_attention_fixed import UNetAttn, UNetAttnSmall
from models.diffusion import GaussianDiffusion, ConditionalDiffusionUNet, DPMSolverPlusPlus


class TestForecastModel(unittest.TestCase):
    """Test cases for deterministic forecast model G_φ."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.channels = 85  # Example: 8 surface + 5*13 pressure + 3 forcing
        self.height = 256
        self.width = 256
        
    def test_model_initialization(self):
        """Test model can be initialized with correct parameters."""
        model = UNetAttn(
            in_ch=self.channels,
            out_ch=self.channels,
            base_features=64,
            use_temporal_encoding=True
        )
        
        # Check model is created
        self.assertIsNotNone(model)
        
        # Check parameter count
        param_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(param_count, 0)
        print(f"Forecast model parameters: {param_count:,}")
    
    def test_forward_pass(self):
        """Test forward pass with random input."""
        model = UNetAttn(
            in_ch=self.channels,
            out_ch=self.channels,
            base_features=32,  # Smaller for testing
            use_temporal_encoding=True
        ).to(self.device)
        
        # Create random input
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        timestamps = torch.randn(self.batch_size).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = model(x, timestamps)
        
        # Check output shape
        expected_shape = (self.batch_size, self.channels, self.height, self.width)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_forward_pass_without_temporal(self):
        """Test forward pass without temporal encoding."""
        model = UNetAttn(
            in_ch=self.channels,
            out_ch=self.channels,
            base_features=32,
            use_temporal_encoding=False
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = UNetAttn(
            in_ch=self.channels,
            out_ch=self.channels,
            base_features=32,
            use_temporal_encoding=True
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        timestamps = torch.randn(self.batch_size).to(self.device)
        target = torch.randn_like(x)
        
        # Forward pass
        output = model(x, timestamps)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}")


class TestDiffusionModel(unittest.TestCase):
    """Test cases for diffusion perturbation model ε_θ."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.channels = 7  # Match our test data (REFC, T2M, D2M, U10, V10, CAPE, CIN)
        self.height = 128  # Smaller for diffusion tests
        self.width = 128
        self.timesteps = 1000
        
    def test_diffusion_process(self):
        """Test Gaussian diffusion process."""
        diffusion = GaussianDiffusion(
            timesteps=self.timesteps,
            beta_schedule="cosine"
        )
        
        # Test beta schedule
        self.assertEqual(len(diffusion.betas), self.timesteps)
        self.assertTrue((diffusion.betas >= 0).all())
        self.assertTrue((diffusion.betas <= 0.02).all())
        
        # Test forward diffusion
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        t = torch.randint(0, self.timesteps, (self.batch_size,))
        
        xt = diffusion.q_sample(x0, t)
        self.assertEqual(xt.shape, x0.shape)
        
    def test_diffusion_unet(self):
        """Test conditional diffusion U-Net."""
        model = ConditionalDiffusionUNet(
            in_channels=self.channels,
            out_channels=self.channels,
            cond_channels=self.channels,
            base_features=32,
            time_emb_dim=128
        ).to(self.device)
        
        # Create inputs
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        timesteps = torch.randint(0, self.timesteps, (self.batch_size,)).to(self.device)
        condition = torch.randn_like(x)
        
        # Forward pass
        with torch.no_grad():
            noise_pred = model(x, timesteps, condition)
        
        # Check output
        self.assertEqual(noise_pred.shape, x.shape)
        self.assertTrue(torch.isfinite(noise_pred).all())
        
    def test_classifier_free_guidance(self):
        """Test model with and without conditioning."""
        model = ConditionalDiffusionUNet(
            in_channels=self.channels,
            out_channels=self.channels,
            cond_channels=self.channels,
            base_features=32
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        timesteps = torch.randint(0, self.timesteps, (self.batch_size,)).to(self.device)
        condition = torch.randn_like(x)
        
        with torch.no_grad():
            # With conditioning
            noise_cond = model(x, timesteps, condition)
            
            # Without conditioning (None)
            noise_uncond = model(x, timesteps, None)
        
        # Both should produce valid outputs
        self.assertEqual(noise_cond.shape, x.shape)
        self.assertEqual(noise_uncond.shape, x.shape)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(noise_cond, noise_uncond))
    
    def test_dpm_solver(self):
        """Test DPM-Solver++ sampling."""
        # Create simple noise prediction function
        def dummy_noise_fn(x, t, c):
            return torch.randn_like(x) * 0.1
        
        # Create alpha schedule
        betas = torch.linspace(0.0001, 0.02, self.timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Initialize solver
        solver = DPMSolverPlusPlus(
            noise_prediction_fn=dummy_noise_fn,
            alphas_cumprod=alphas_cumprod,
            num_timesteps=self.timesteps,
            solver_order=2
        )
        
        # Test sampling
        x_T = torch.randn(self.batch_size, self.channels, self.height, self.width)
        condition = torch.randn_like(x_T)
        
        x_0 = solver.sample(
            x_T,
            num_steps=50,
            condition=condition,
            guidance_weight=0.1
        )
        
        # Check output
        self.assertEqual(x_0.shape, x_T.shape)
        self.assertTrue(torch.isfinite(x_0).all())
    
    def test_training_loss(self):
        """Test diffusion training loss computation."""
        diffusion = GaussianDiffusion(
            timesteps=self.timesteps,
            beta_schedule="cosine"
        ).to(self.device)
        
        model = ConditionalDiffusionUNet(
            in_channels=self.channels,
            out_channels=self.channels,
            cond_channels=self.channels,
            base_features=32
        ).to(self.device)
        
        # Create batch
        x_start = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        condition = torch.randn_like(x_start)
        
        # Compute loss
        losses = diffusion.training_losses(
            model,
            x_start,
            condition=condition,
            dropout_prob=0.1
        )
        
        # Check losses
        self.assertIn('loss', losses)
        self.assertIn('loss_conditional', losses)
        self.assertIn('loss_unconditional', losses)
        
        # All losses should be finite
        for key, loss in losses.items():
            self.assertTrue(torch.isfinite(loss).all(), f"{key} is not finite")


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_forecast_diffusion_integration(self):
        """Test forecast model output can be used as diffusion condition."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        channels = 20  # Smaller for integration test
        batch_size = 1
        height, width = 64, 64
        
        # Create models
        forecast_model = UNetAttnSmall(
            in_ch=channels,
            out_ch=channels,
            use_temporal_encoding=True
        ).to(device)
        
        diffusion_model = ConditionalDiffusionUNet(
            in_channels=channels,
            out_channels=channels,
            cond_channels=channels,
            base_features=32
        ).to(device)
        
        # Create input
        x0 = torch.randn(batch_size, channels, height, width).to(device)
        timestamps = torch.tensor([100.0]).to(device)  # 100 hours
        
        # Get forecast
        with torch.no_grad():
            forecast = forecast_model(x0, timestamps)
        
        # Use forecast as condition for diffusion
        timesteps = torch.tensor([500]).to(device)  # Diffusion timestep
        noise = torch.randn_like(x0)
        
        with torch.no_grad():
            noise_pred = diffusion_model(noise, timesteps, forecast)
        
        # Check outputs
        self.assertEqual(forecast.shape, x0.shape)
        self.assertEqual(noise_pred.shape, x0.shape)
        self.assertTrue(torch.isfinite(forecast).all())
        self.assertTrue(torch.isfinite(noise_pred).all())


if __name__ == '__main__':
    unittest.main()