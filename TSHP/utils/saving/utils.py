import random
import torch
import time
import os

def safe_write(obj, path,
               timeout_s=120):  # I'm having trouble figuring out a safe way to use torch.save() without race conditions.
    # It's a bit hacky but i'm just going to use random seeds and hope that 0.01% chance that the tmp name overlaps doesn't happen
    tmp_path = path + str(random.randint(0, 99999)).zfill(5)
    
    # rather than corrupting, one file should simply replace the other
    torch.save(obj, tmp_path)
    i = 0
    while i < timeout_s:
        i += 1
        try:
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)
            break
        except:
            time.sleep(1)
    if os.path.exists(tmp_path): # file should have been moved, but in the case where it somehow wasn't moved, delete it.
        print(f"Failed to write '{path}'")
        os.remove(tmp_path)

def try_torchload(path, **kwargs):
    try:
        return torch.load(path, **kwargs)
    except Exception:
        if kwargs.get('print_warning', True):
            print(f"Failed to torch.load('{path}')")
        return None