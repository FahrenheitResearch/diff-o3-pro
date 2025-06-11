import torch
from models.unet_attention import UNetAttn
from hrrr_dataset.hrrr_data import HRRRDataset
from pathlib import Path

# Load one sample
ds = HRRRDataset(
    Path("data/zarr/training_data/hrrr.zarr"),
    ["REFC", "T2M", "D2M", "U10", "V10", "CAPE", "CIN"],
    1,
    Path("data/stats.json")
)

print(f"Dataset length: {len(ds)}")

# Get one sample
x, y = ds[0]
x = x.unsqueeze(0)  # Add batch dimension
print(f"Input shape: {x.shape}")
print(f"Target shape: {y.shape}")

# Create model with correct input/output channels
n_channels = 7  # We have 7 variables
model = UNetAttn(n_channels, n_channels, nf=32)  # Reduced nf for testing

print("\nModel created successfully")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Try forward pass with hooks to debug
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[name] = output.shape
        return output
    return hook

# Register hooks on key layers
model.d1.register_forward_hook(hook_fn('d1'))
model.d2.register_forward_hook(hook_fn('d2'))
model.d3.register_forward_hook(hook_fn('d3'))
model.bridge.register_forward_hook(hook_fn('bridge'))

try:
    with torch.no_grad():
        output = model(x)
    print(f"\nForward pass successful!")
    print(f"Output shape: {output.shape}")
    
    print("\nIntermediate shapes:")
    for name, shape in activations.items():
        print(f"  {name}: {shape}")
        
except Exception as e:
    print(f"\nError during forward pass: {e}")
    print("\nActivations before error:")
    for name, shape in activations.items():
        print(f"  {name}: {shape}")