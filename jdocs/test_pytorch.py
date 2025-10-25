import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # Test tensor operation
    x = torch.rand(5, 3).cuda()
    print(f"\nTest tensor on GPU:\n{x}")
    
    # Simple computation test
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.matmul(a, b)
    
    print("\n✓ GPU is working correctly!")
    print(f"✓ Successfully performed matrix multiplication on {torch.cuda.get_device_name(0)}")
else:
    print("\n✗ CUDA is not available. Check your installation.")