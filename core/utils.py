"""
Utility functions to replace dassl.utils
"""
import os
import sys
import time
import random
import numpy as np
import torch
import logging


class _Tee:
    """Tee stdout/stderr to both console and file."""
    def __init__(self, stream_main, stream_side):
        self.stream_main = stream_main
        self.stream_side = stream_side
        self._is_tee = True
    def write(self, data):
        self.stream_main.write(data)
        try:
            self.stream_side.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            self.stream_main.flush()
            self.stream_side.flush()
        except Exception:
            pass


def setup_logger(output_dir, name="sentiment_analysis"):
    """Setup logger to output to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "log.txt")
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # Tee print() to file as well
        try:
            if not getattr(sys.stdout, "_is_tee", False):
                sys.stdout = _Tee(sys.stdout, open(log_file, mode='a', encoding='utf-8'))
            if not getattr(sys.stderr, "_is_tee", False):
                sys.stderr = _Tee(sys.stderr, open(log_file, mode='a', encoding='utf-8'))
        except Exception:
            pass
        print(f"Logging to file: {log_file}")
    
    return logger


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_env_info():
    """Collect environment information."""
    info = []
    info.append(f"PyTorch version: {torch.__version__}")
    info.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        info.append(f"CUDA version: {torch.version.cuda}")
        info.append(f"Number of GPUs: {torch.cuda.device_count()}")
    info.append(f"Python version: {sys.version}")
    return "\n".join(info)


def load_checkpoint(fpath):
    """Load checkpoint from file."""
    if not os.path.exists(fpath):
        raise FileNotFoundError(f'Checkpoint not found at "{fpath}"')
    
    print(f"Loading checkpoint from {fpath}")
    load_kwargs = {"map_location": "cpu"}
    try:
        checkpoint = torch.load(fpath, weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(fpath, **load_kwargs)
    return checkpoint


def save_checkpoint(state, save_dir, is_best=False, model_name="model"):
    """Save checkpoint to file."""
    os.makedirs(save_dir, exist_ok=True)
    
    fpath = os.path.join(save_dir, f"{model_name}.pth.tar")
    torch.save(state, fpath)
    print(f"Checkpoint saved to {fpath}")
    
    if is_best:
        best_fpath = os.path.join(save_dir, f"{model_name}-best.pth.tar")
        torch.save(state, best_fpath)
        print(f"Best checkpoint saved to {best_fpath}")


def load_pretrained_weights(model, weight_path):
    """Load pretrained weights to model."""
    checkpoint = load_checkpoint(weight_path)
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {weight_path}")


class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_if_missing(dirname):
    """Create dirname if it does not exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
