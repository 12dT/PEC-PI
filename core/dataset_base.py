"""
Base dataset classes to replace dassl.data.datasets
"""
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class Datum:
    """Data instance for sentiment analysis with text and image."""
    
    def __init__(self, impath="", label=0, classname="", text="", **kwargs):
        self.impath = impath
        self.label = label
        self.classname = classname
        self.text = text
        
        # Store additional attributes (e.g., caption_tone, caption_content, caption_emotion)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return f"Datum(impath={self.impath}, label={self.label}, classname={self.classname})"


class DatasetBase:
    """Base class for datasets."""
    
    dataset_dir = ""
    classnames = []
    
    def __init__(self, train_x=None, val=None, test=None):
        self._train_x = train_x if train_x is not None else []
        self._val = val if val is not None else []
        self._test = test if test is not None else []
        self._num_classes = len(self.classnames)
    
    @property
    def train_x(self):
        return self._train_x
    
    @property
    def val(self):
        return self._val
    
    @property
    def test(self):
        return self._test
    
    @property
    def num_classes(self):
        return self._num_classes
    
    def generate_fewshot_dataset(self, data_source, num_shots=-1, repeat=False):
        """Generate a few-shot dataset."""
        if num_shots < 1:
            return data_source
        
        from collections import defaultdict
        import random
        
        output = []
        tracker = defaultdict(list)
        
        for item in data_source:
            tracker[item.label].append(item)
        
        for label, items in tracker.items():
            if len(items) >= num_shots:
                sampled = random.sample(items, num_shots)
            else:
                if repeat:
                    sampled = random.choices(items, k=num_shots)
                else:
                    sampled = items
            output.extend(sampled)
        
        return output


class DatasetWrapper(Dataset):
    """Wrapper for PyTorch Dataset."""
    
    def __init__(self, data_source: List[Datum], transform=None, is_train=True):
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        item = self.data_source[idx]
        
        # Load and transform image
        try:
            img = Image.open(item.impath).convert("RGB")
        except Exception as e:
            print(f"Error loading image {item.impath}: {e}")
            # Return a blank image
            img = Image.new("RGB", (224, 224))
        
        if self.transform is not None:
            img = self.transform(img)
        
        output = {
            "img": img,
            "label": item.label,
            "text": item.text if hasattr(item, "text") else "",
            "impath": item.impath,
            "classname": item.classname
        }
        
        # Add three-stage caption information (CRITICAL for three-stage prompting!)
        for attr in ["caption_tone", "caption_content", "caption_emotion", 
                     "text_label", "image_label"]:
            if hasattr(item, attr):
                value = getattr(item, attr)
                output[attr] = value if value else ""  # Ensure not None
        
        return output


# Registry for datasets
DATASET_REGISTRY = {}

def register_dataset(name):
    """Decorator to register a dataset class."""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def build_dataset(cfg):
    """Build dataset from config."""
    dataset_name = cfg.DATASET.NAME
    
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Available datasets: {available}"
        )
    
    dataset_class = DATASET_REGISTRY[dataset_name]
    return dataset_class(cfg)
