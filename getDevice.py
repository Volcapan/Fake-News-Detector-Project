import torch

def get_device_func():
    return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"