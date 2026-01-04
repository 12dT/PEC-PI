import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

from core.utils import setup_logger, set_random_seed, collect_env_info
from core.config import get_cfg_default
from core.trainer_base import build_trainer

# Import datasets to register them
import datasets.mvsa_single  # MVSA-Single sentiment dataset

# Import trainers to register them
import trainers.maple

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables for sentiment analysis with MaPLe.
    """
    from yacs.config import CfgNode as CN

    # Config for MaPLe - optimized for sentiment analysis
    cfg.TRAINER.NAME = "MaPLe"
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 4  # number of context vectors (increased for better sentiment representation)
    cfg.TRAINER.MAPLE.CTX_INIT = "a sentiment of"  # initialization words for sentiment task
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    
    # Two-stage training strategy
    cfg.DATASET.TWO_STAGE = False  # Enable two-stage training
    cfg.DATASET.STAGE = 1  # 1 or 2

    # Caption contrastive loss to leverage caption/text annotations
    cfg.TRAINER.MAPLE.CAPTION_LOSS = CN()
    cfg.TRAINER.MAPLE.CAPTION_LOSS.ENABLE = True
    cfg.TRAINER.MAPLE.CAPTION_LOSS.WEIGHT = 0.5
    cfg.TRAINER.MAPLE.CAPTION_LOSS.TRUNCATE = True
    cfg.TRAINER.MAPLE.CAPTION_LOSS.TEMP = 1.0  # additional temperature for caption InfoNCE (1.0 = no change)
    cfg.TRAINER.MAPLE.CAPTION_LOSS.RAMP_EPOCHS = 5  # ramp-up epochs for caption loss weight
    cfg.TRAINER.MAPLE.CAPTION_LOSS.GATING = CN()
    cfg.TRAINER.MAPLE.CAPTION_LOSS.GATING.ENABLE = False
    cfg.TRAINER.MAPLE.CAPTION_LOSS.GATING.THRESH = 0.2  # cosine similarity threshold in [−1,1]
    # Three-stage supervision (tone/content/emotion)
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED = CN()
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ENABLE = True
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.WEIGHTS = [0.2, 0.2, 0.6]  # tone, content, emotion
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ADAPTIVE = CN()
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ADAPTIVE.ENABLE = True
    # Hierarchical supervision: introduce stages progressively
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL = CN()
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL.ENABLE = False
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL.STAGE1_EPOCHS = 10  # tone only
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.HIERARCHICAL.STAGE2_EPOCHS = 30  # tone + content
    # Stage 3 (tone + content + emotion) starts after STAGE2_EPOCHS

    # Test-time augmentation
    cfg.TEST.TTA = CN()
    cfg.TEST.TTA.ENABLE = False
    cfg.TEST.TTA.HFLIP = True
    
    # Text encoder fine-tuning
    cfg.TRAINER.MAPLE.FINETUNE_TEXT = CN()
    cfg.TRAINER.MAPLE.FINETUNE_TEXT.ENABLE = False
    cfg.TRAINER.MAPLE.FINETUNE_TEXT.LAYERS = 2  # Unfreeze last N layers
    cfg.TRAINER.MAPLE.FINETUNE_TEXT.LR_MULT = 0.1  # LR multiplier for text encoder
    
    # Separate text contrastive loss (sample text independent from caption)
    cfg.TRAINER.MAPLE.TEXT_LOSS = CN()
    cfg.TRAINER.MAPLE.TEXT_LOSS.ENABLE = False
    cfg.TRAINER.MAPLE.TEXT_LOSS.WEIGHT = 0.2
    
    # Dual-stream fusion for true multimodal classification
    cfg.TRAINER.MAPLE.DUAL_STREAM = CN()
    cfg.TRAINER.MAPLE.DUAL_STREAM.ENABLE = False
    cfg.TRAINER.MAPLE.DUAL_STREAM.ADAPTIVE_WEIGHT = True  # Learn image/text weights
    cfg.TRAINER.MAPLE.DUAL_STREAM.IMAGE_WEIGHT = 0.6  # Fixed weight if not adaptive
    cfg.TRAINER.MAPLE.DUAL_STREAM.TEXT_WEIGHT = 0.4  # Fixed weight for text lado if not adaptive
    cfg.TRAINER.MAPLE.DUAL_STREAM.AUX_LOSS_WEIGHT = 0.3  # Auxiliary loss for each stream
    cfg.TRAINER.MAPLE.DUAL_STREAM.FUSION_TYPE = "weighted"  # "weighted", "gated", "fusion_only", "diff_attn"
    cfg.TRAINER.MAPLE.DUAL_STREAM.DIFF_ATTN = CN()
    cfg.TRAINER.MAPLE.DUAL_STREAM.DIFF_ATTN.NUM_HEADS = 8
    cfg.TRAINER.MAPLE.DUAL_STREAM.DIFF_ATTN.LAMBDA_NEG = 0.5
    cfg.TRAINER.MAPLE.DUAL_STREAM.SEPARATE_TEXT_ENCODER = True  # Use separate encoder for sample text  # Suppression coefficient
    
    # Class-balanced sampling
    cfg.DATALOADER.TRAIN_X.CLASS_BALANCED = False  # Enable class-balanced sampling


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
