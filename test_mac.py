import torch
if torch.backends.mps.is_available():
    print("MPS is available! Your Mac's GPU is ready.")
else:
    print("MPS not found. You will train on the CPU (which is still totally fine!).")