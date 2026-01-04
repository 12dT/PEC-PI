"""
Base trainer class to replace dassl.engine.TrainerX
"""
import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from .utils import AverageMeter, save_checkpoint, load_checkpoint
from .optim import compute_accuracy
from .dataset_base import DatasetWrapper


class SimpleTrainer:
    """Base trainer for multimodal sentiment analysis."""
    
    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.batch_idx = 0
        self.num_batches = 0
        self.best_result = -1
        self.best_result_f1 = -1
        self._last_val_metrics = None
    
    def register_model(self, name="model", model=None, optim=None, sched=None):
        """Register model, optimizer, and scheduler."""
        if model is not None:
            self._models[name] = model
        if optim is not None:
            self._optims[name] = optim
        if sched is not None:
            self._scheds[name] = sched
    
    def get_model_names(self, names=None):
        """Get model names."""
        names_real = list(self._models.keys())
        if names is not None:
            names_real = [name for name in names_real if name in names]
        return names_real
    
    def save_model(self, epoch, directory, is_best=False):
        """Save model checkpoint."""
        names = self.get_model_names()
        
        for name in names:
            model = self._models[name]
            optim = self._optims.get(name)
            sched = self._scheds.get(name)
            
            state = {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer": optim.state_dict() if optim else None,
                "scheduler": sched.state_dict() if sched else None
            }
            
            save_dir = os.path.join(directory, name)
            save_checkpoint(state, save_dir, is_best=is_best, model_name="model")
    
    def resume_model_if_exist(self, directory):
        """Resume model from checkpoint if exists."""
        names = self.get_model_names()
        file_missing = False
        
        for name in names:
            path = os.path.join(directory, name)
            if not os.path.exists(path):
                file_missing = True
                break
        
        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0
        
        print(f"Found checkpoint in {directory}, resume training")
        
        for name in names:
            path = os.path.join(directory, name, "model.pth.tar")
            
            checkpoint = load_checkpoint(path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            print(f"Loading model weights from epoch {epoch}")
            self._models[name].load_state_dict(state_dict)
            
            if self._optims.get(name) and checkpoint.get("optimizer"):
                self._optims[name].load_state_dict(checkpoint["optimizer"])
            
            if self._scheds.get(name) and checkpoint.get("scheduler"):
                self._scheds[name].load_state_dict(checkpoint["scheduler"])
        
        return epoch
    
    def set_model_mode(self, mode="train", names=None):
        """Set model mode (train or eval)."""
        names = self.get_model_names(names)
        
        for name in names:
            if mode == "train":
                self._models[name].train()
            else:
                self._models[name].eval()
    
    def update_lr(self, names=None):
        """Update learning rate."""
        names = self.get_model_names(names)
        
        for name in names:
            if self._scheds.get(name) is not None:
                self._scheds[name].step()
    
    def detect_anomaly(self, loss):
        """Detect anomaly in loss."""
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")
    
    def model_zero_grad(self, names=None):
        """Zero gradients of models."""
        names = self.get_model_names(names)
        for name in names:
            if self._optims.get(name) is not None:
                self._optims[name].zero_grad()
    
    def model_backward(self, loss):
        """Backward pass."""
        self.detect_anomaly(loss)
        loss.backward()
    
    def model_update(self, names=None):
        """Update model parameters."""
        names = self.get_model_names(names)
        for name in names:
            if self._optims.get(name) is not None:
                self._optims[name].step()


class TrainerX(SimpleTrainer):
    """Trainer with data loading and training loop."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
        
        # Build data loader
        self.build_data_loader()
        
        # Build model
        self.build_model()
        
        # Resume model if checkpoint exists
        if cfg.RESUME:
            self.start_epoch = self.resume_model_if_exist(cfg.RESUME)
        
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize stage boundaries and best results for each stage
        self._init_stage_boundaries()
    
    def build_data_loader(self):
        """Build data loader. To be implemented by subclass."""
        # Create transforms
        normalize = transforms.Normalize(
            mean=self.cfg.INPUT.PIXEL_MEAN,
            std=self.cfg.INPUT.PIXEL_STD
        )
        
        train_transform = transforms.Compose([
            transforms.Resize(self.cfg.INPUT.SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(self.cfg.INPUT.SIZE[0], padding=self.cfg.INPUT.CROP_PADDING),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(self.cfg.INPUT.SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.cfg.INPUT.SIZE[0]),
            transforms.ToTensor(),
            normalize
        ])
        
        # Build dataset (to be loaded from dataset registry)
        from .dataset_base import build_dataset
        dataset = build_dataset(self.cfg)
        self.dm = type('DataManager', (), {
            'dataset': dataset,
            'num_classes': dataset.num_classes
        })()
        
        # Build data loaders
        train_dataset = DatasetWrapper(dataset.train_x, transform=train_transform, is_train=True)
        
        # Class-balanced sampling if enabled
        if self.cfg.DATALOADER.TRAIN_X.CLASS_BALANCED:
            from collections import Counter
            import numpy as np
            # Count samples per class
            labels = [item.label for item in dataset.train_x]
            class_counts = Counter(labels)
            num_classes = len(class_counts)
            print(f"Original class distribution: {dict(class_counts)}")
            
            # Compute sample weights (inverse frequency)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[item.label] for item in dataset.train_x]
            sample_weights = torch.DoubleTensor(sample_weights)
            
            # WeightedRandomSampler
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            print(f"Using class-balanced sampling with weights: {class_weights}")
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                sampler=sampler,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                drop_last=True
            )
        else:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                shuffle=True,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                drop_last=True
            )
        
        val_dataset = DatasetWrapper(dataset.val, transform=test_transform, is_train=False)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS
        )
        
        test_dataset = DatasetWrapper(dataset.test, transform=test_transform, is_train=False)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS
        )
        
        print(f"Train: {len(train_dataset)} samples")
        print(f"Val: {len(val_dataset)} samples")
        print(f"Test: {len(test_dataset)} samples")
    
    def _init_stage_boundaries(self):
        """Initialize stage boundaries based on config."""
        # Check if hierarchical stages are enabled
        if (hasattr(self.cfg, 'TRAINER') and 
            hasattr(self.cfg.TRAINER, 'MAPLE') and
            hasattr(self.cfg.TRAINER.MAPLE, 'CAPTION_LOSS') and
            hasattr(self.cfg.TRAINER.MAPLE.CAPTION_LOSS, 'STAGED') and
            hasattr(self.cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED, 'HIERARCHICAL') and
            self.cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL.ENABLE):
            # Use hierarchical stage boundaries
            stage1_end = self.cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL.STAGE1_EPOCHS
            stage2_end = self.cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL.STAGE2_EPOCHS
            self.stage_boundaries = [0, stage1_end, stage2_end, self.max_epoch]
            print(f"Using hierarchical stage boundaries: Stage1 (1-{stage1_end}), Stage2 ({stage1_end+1}-{stage2_end}), Stage3 ({stage2_end+1}-{self.max_epoch})")
        else:
            # Use RAMP_EPOCHS to divide into 3 stages
            ramp_epochs = 15  # default
            if (hasattr(self.cfg, 'TRAINER') and 
                hasattr(self.cfg.TRAINER, 'MAPLE') and
                hasattr(self.cfg.TRAINER.MAPLE, 'CAPTION_LOSS') and
                hasattr(self.cfg.TRAINER.MAPLE.CAPTION_LOSS, 'RAMP_EPOCHS')):
                ramp_epochs = self.cfg.TRAINER.MAPLE.CAPTION_LOSS.RAMP_EPOCHS
            
            # Divide RAMP_EPOCHS into 3 equal segments
            segment = max(1, ramp_epochs // 3)
            stage1_end = segment
            stage2_end = 2 * segment
            # Stage 3 ends at RAMP_EPOCHS
            stage3_end = ramp_epochs
            self.stage_boundaries = [0, stage1_end, stage2_end, stage3_end]
            print(f"Using RAMP_EPOCHS-based stage boundaries (RAMP_EPOCHS={ramp_epochs}): Stage1 (1-{stage1_end}), Stage2 ({stage1_end+1}-{stage2_end}), Stage3 ({stage2_end+1}-{stage3_end})")
        
        # Initialize best results for each stage (using accuracy)
        self.stage_best_results = [-1.0] * 3  # [stage1_best, stage2_best, stage3_best]
        self.stage_best_f1 = [-1.0] * 3  # [stage1_best_f1, stage2_best_f1, stage3_best_f1]
    
    def _get_current_stage(self, epoch):
        """Get the current stage index (0, 1, or 2) for the given epoch.
        Returns None if epoch is beyond all stages."""
        if epoch < self.stage_boundaries[1]:
            return 0  # Stage 1
        elif epoch < self.stage_boundaries[2]:
            return 1  # Stage 2
        elif epoch < self.stage_boundaries[3]:
            return 2  # Stage 3
        else:
            return None  # Beyond all stages
    
    def build_model(self):
        """Build model. To be implemented by subclass."""
        raise NotImplementedError
    
    def train(self):
        """Training loop."""
        self.start_time = time.time()
        self.before_train()
        
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        
        self.after_train()
    
    def before_train(self):
        """Operations before training."""
        print("Start training")
        print(f"Output directory: {self.output_dir}")
    
    def after_train(self):
        """Operations after training."""
        print("Finished training")
        
        # Test on best model
        print("Test on best model")
        self.load_model(self.output_dir)
        self.test()
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=int(elapsed)))
        print(f"Elapsed time: {elapsed}")
    
    def before_epoch(self):
        """Operations before each epoch."""
        pass
    
    def after_epoch(self):
        """Operations after each epoch."""
        # Print dual-stream weights if enabled (only for weighted mode)
        if hasattr(self, 'cfg') and hasattr(self.cfg.TRAINER, 'MAPLE'):
            if self.cfg.TRAINER.MAPLE.DUAL_STREAM.ENABLE and self.cfg.TRAINER.MAPLE.DUAL_STREAM.ADAPTIVE_WEIGHT:
                if self.cfg.TRAINER.MAPLE.DUAL_STREAM.FUSION_TYPE == "weighted" and hasattr(self.model, 'fusion_weight_logits'):
                    with torch.no_grad():
                        weights = torch.softmax(torch.cat([self.model.fusion_weight_logits, torch.zeros(1, device=self.model.fusion_weight_logits.device)]), dim=0)
                        print(f"Dual-stream weights @epoch{self.epoch+1}: img={weights[0].item():.3f}, text={weights[1].item():.3f}, fused={weights[2].item():.3f}")
            # Print caption staged weights if enabled
            if self.cfg.TRAINER.MAPLE.CAPTION_LOSS.ENABLE and self.cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ENABLE:
                if self.cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ADAPTIVE.ENABLE:
                    if hasattr(self.model, 'prompt_learner') and hasattr(self.model.prompt_learner, 'staged_weight_logits'):
                        with torch.no_grad():
                            cap_w = torch.softmax(self.model.prompt_learner.staged_weight_logits, dim=0)
                            stage_info = f"Caption staged weights @epoch{self.epoch+1}: tone={cap_w[0].item():.3f}, content={cap_w[1].item():.3f}, emotion={cap_w[2].item():.3f}"
                            # Add hierarchical stage indicator
                            if self.cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL.ENABLE:
                                stage1 = self.cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL.STAGE1_EPOCHS
                                stage2 = self.cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL.STAGE2_EPOCHS
                                if self.epoch < stage1:
                                    stage_info += " [Stage 1: Perception/tone only]"
                                elif self.epoch < stage2:
                                    stage_info += " [Stage 2: +Semantics/content]"
                                else:
                                    stage_info += " [Stage 3: +Affect/emotion]"
                            print(stage_info)
        
        # Evaluate on validation set
        print(f"\nValidation at epoch {self.epoch + 1}")
        # Run validation and cache metrics
        val_acc = self.test(split="val")
        val_f1 = None
        if self._last_val_metrics is not None:
            val_f1 = self._last_val_metrics.get("f1_macro", None)
        
        # Save model
        is_best = False
        if val_acc > self.best_result:
            self.best_result = val_acc
            is_best = True
        
        self.save_model(self.epoch, self.output_dir, is_best=is_best)

        # Save best-f1 checkpoint separately as model-best-f1.pth.tar
        if val_f1 is not None and val_f1 > self.best_result_f1:
            self.best_result_f1 = val_f1
            names = self.get_model_names()
            for name in names:
                model = self._models[name]
                optim = self._optims.get(name)
                sched = self._scheds.get(name)
                state = {
                    "state_dict": model.state_dict(),
                    "epoch": self.epoch + 1,
                    "optimizer": optim.state_dict() if optim else None,
                    "scheduler": sched.state_dict() if sched else None
                }
                save_dir = os.path.join(self.output_dir, name)
                # use model_name to produce model-best-f1.pth.tar
                save_checkpoint(state, save_dir, is_best=False, model_name="model-best-f1")
        
        # Save best checkpoint for current stage
        current_stage = self._get_current_stage(self.epoch)
        
        # Only save stage checkpoints if we're within a defined stage
        if current_stage is not None:
            # Check if current epoch has best accuracy for this stage
            if val_acc > self.stage_best_results[current_stage]:
                self.stage_best_results[current_stage] = val_acc
                names = self.get_model_names()
                for name in names:
                    model = self._models[name]
                    optim = self._optims.get(name)
                    sched = self._scheds.get(name)
                    state = {
                        "state_dict": model.state_dict(),
                        "epoch": self.epoch + 1,
                        "optimizer": optim.state_dict() if optim else None,
                        "scheduler": sched.state_dict() if sched else None
                    }
                    save_dir = os.path.join(self.output_dir, name)
                    # Save as model-stage{N}-best.pth.tar
                    save_checkpoint(state, save_dir, is_best=False, model_name=f"model-stage{current_stage+1}-best")
                    print(f"Saved best checkpoint for Stage {current_stage+1} (epoch {self.epoch+1}, acc={val_acc:.2f}%)")
            
            # Check if current epoch has best F1 for this stage
            if val_f1 is not None and val_f1 > self.stage_best_f1[current_stage]:
                self.stage_best_f1[current_stage] = val_f1
                names = self.get_model_names()
                for name in names:
                    model = self._models[name]
                    optim = self._optims.get(name)
                    sched = self._scheds.get(name)
                    state = {
                        "state_dict": model.state_dict(),
                        "epoch": self.epoch + 1,
                        "optimizer": optim.state_dict() if optim else None,
                        "scheduler": sched.state_dict() if sched else None
                    }
                    save_dir = os.path.join(self.output_dir, name)
                    # Save as model-stage{N}-best-f1.pth.tar
                    save_checkpoint(state, save_dir, is_best=False, model_name=f"model-stage{current_stage+1}-best-f1")
                    print(f"Saved best F1 checkpoint for Stage {current_stage+1} (epoch {self.epoch+1}, f1={val_f1:.2f}%)")
        
        # Update learning rate
        self.update_lr()
    
    def run_epoch(self):
        """Run one epoch."""
        self.set_model_mode("train")
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        self.num_batches = len(self.train_loader)
        end = time.time()
        
        for self.batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary["loss"])
            
            # Print progress
            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_remain = self.num_batches - self.batch_idx - 1
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(
                    f"Epoch [{self.epoch + 1}/{self.max_epoch}] "
                    f"Batch [{self.batch_idx + 1}/{self.num_batches}] "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                    f"ETA {eta}"
                )
            
            end = time.time()
    
    def forward_backward(self, batch):
        """Forward and backward pass. To be implemented by subclass."""
        raise NotImplementedError
    
    def parse_batch_train(self, batch):
        """Parse batch data."""
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    @torch.no_grad()
    def test(self, split="test"):
        """Test the model."""
        self.set_model_mode("eval")
        
        if split == "val":
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader
        
        print(f"Evaluating on {split} set")
        
        total = 0
        correct = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(data_loader):
            batch_result = self.parse_batch_train(batch)
            if len(batch_result) == 3:  # dual-stream mode: (input, label, sample_text_tokens)
                input, label, sample_text_tokens = batch_result
                output = self.model_inference(input, sample_text_tokens)
            else:  # original mode: (input, label)
                input, label = batch_result
                output = self.model_inference(input)
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            # Collect predictions and labels for F1 score
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())
        
        accuracy = 100.0 * correct / total
        
        # Calculate F1 scores
        from .optim import compute_f1_score
        f1_macro, f1_weighted, f1_per_class = compute_f1_score(all_preds, all_labels, num_classes=3)
        
        print(f"{split.capitalize()} Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  F1-Macro: {f1_macro:.2f}%")
        print(f"  F1-Weighted: {f1_weighted:.2f}%")
        print(f"  F1-PerClass: Negative={f1_per_class[0]:.2f}%, Neutral={f1_per_class[1]:.2f}%, Positive={f1_per_class[2]:.2f}%")
        
        # Plot confusion matrix for test set
        if split == "test":
            self._plot_confusion_matrix(all_labels, all_preds, split)
        
        # Cache last val metrics for saving policies
        if split == "val":
            self._last_val_metrics = {
                "acc": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "f1_per_class": f1_per_class,
            }
        
        return accuracy
    
    def _plot_confusion_matrix(self, y_true, y_pred, split="test"):
        """Plot and save confusion matrix."""
        try:
            # Class names
            class_names = ['Negative', 'Neutral', 'Positive']
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Count'})
            plt.title(f'Confusion Matrix ({split.capitalize()} Set)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Save to output directory
            save_path = os.path.join(self.output_dir, f'confusion_matrix_{split}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to {save_path}")
            
            # Also save normalized version (percentages)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Percentage (%)'})
            plt.title(f'Confusion Matrix - Normalized ({split.capitalize()} Set)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            save_path_norm = os.path.join(self.output_dir, f'confusion_matrix_{split}_normalized.png')
            plt.savefig(save_path_norm, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Normalized confusion matrix saved to {save_path_norm}")
            
        except Exception as e:
            print(f"Warning: Failed to plot confusion matrix: {e}")
    
    def model_inference(self, input):
        """Model inference. To be implemented by subclass."""
        raise NotImplementedError
    
    def load_model(self, directory, epoch=None):
        """Load model from directory."""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        
        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        
        if epoch is not None:
            model_file = f"model.pth.tar-{epoch}"
        
        for name in names:
            model_path = os.path.join(directory, name, model_file)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')
            
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch_loaded = checkpoint["epoch"]
            
            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch_loaded})')
            self._models[name].load_state_dict(state_dict, strict=False)

    def load_model_best_f1(self, directory):
        """Load model weights tagged as best_f1 (model-best-f1.pth.tar)."""
        if not directory:
            print("Note that load_model_best_f1() is skipped as no pretrained model is given")
            return
        names = self.get_model_names()
        for name in names:
            model_path = os.path.join(directory, name, "model-best-f1.pth.tar")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch_loaded = checkpoint["epoch"]
            print(f'Loading best-f1 weights to {name} from "{model_path}" (epoch = {epoch_loaded})')
            self._models[name].load_state_dict(state_dict, strict=False)


# Registry for trainers
TRAINER_REGISTRY = {}

def register_trainer(name=None):
    """Decorator to register a trainer class."""
    def decorator(cls):
        trainer_name = name if name is not None else cls.__name__
        TRAINER_REGISTRY[trainer_name] = cls
        return cls
    return decorator


def build_trainer(cfg):
    """Build trainer from config."""
    trainer_name = cfg.TRAINER.NAME
    
    if trainer_name not in TRAINER_REGISTRY:
        available = ", ".join(TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Trainer '{trainer_name}' not found. "
            f"Available trainers: {available}"
        )
    
    trainer_class = TRAINER_REGISTRY[trainer_name]
    return trainer_class(cfg)
