import torch

# get device
def get_device():
    device = torch.device("cuda")
    return device

# check local gpu
def check_gpu(pb = True):
    cuda_available = torch.cuda.is_available()
    print(f"GPU available {cuda_available}")

    if not cuda_available:
        print("No GPU")
        exit(0)
    if pb:
        print(f"GPU number:{torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"current GPU index:{current_device}")
        # GPU name
        print(f"GPU name:{torch.cuda.get_device_name(current_device)}")
        # CUDA version
        print(f"CUDA version:{torch.version.cuda}")
        # torch version
        print(f"torch version:{torch.__version__}")



