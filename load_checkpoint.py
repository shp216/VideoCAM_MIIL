import torch
import os

def save_checkpoint(path, phi, optimizer, lr_scheduler, epoch, step, accelerator):
    if accelerator.is_main_process:
        checkpoint_data = {
            'model_state_dict': accelerator.unwrap_model(phi).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
        }
        torch.save(checkpoint_data, path)
        print(f"[INFO] Checkpoint saved at {path}")
    accelerator.wait_for_everyone()  # Synchronize processes


def load_checkpoint(path, phi, optimizer, lr_scheduler, accelerator):
    if os.path.exists(path):
        print(f"[INFO] Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location='cpu')  # Load to CPU first
        accelerator.unwrap_model(phi).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        print(f"[INFO] Successfully loaded checkpoint from epoch {epoch}, step {step}")
        return epoch, step
    else:
        print(f"[INFO] No checkpoint found at {path}. Starting from scratch.")
        return 0, 0  # Start from epoch 0, step 0 if no checkpoint is found