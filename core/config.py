"""
Configuration system to replace dassl.config
"""
from yacs.config import CfgNode as CN


def get_cfg_default():
    """Get a yacs CfgNode object with default values."""
    cfg = CN()
    
    # Model settings
    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
    cfg.MODEL.INIT_WEIGHTS = ""
    
    # Dataset settings
    cfg.DATASET = CN()
    cfg.DATASET.NAME = ""
    cfg.DATASET.ROOT = "datasets"
    cfg.DATASET.NUM_SHOTS = -1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.SOURCE_DOMAINS = ()
    cfg.DATASET.TARGET_DOMAINS = ()
    
    # Dataloader settings
    cfg.DATALOADER = CN()
    cfg.DATALOADER.TRAIN_X = CN()
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 32
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = 0
    cfg.DATALOADER.TRAIN_X.N_INS = 16
    cfg.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"
    cfg.DATALOADER.TRAIN_X.SAME_AS_X = True  # For compatibility
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = 32
    cfg.DATALOADER.TEST.SAMPLER = "SequentialSampler"
    cfg.DATALOADER.TEST.N_DOMAIN = 0
    cfg.DATALOADER.TEST.N_INS = 16
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Input settings
    cfg.INPUT = CN()
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.INTERPOLATION = "bicubic"
    cfg.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
    cfg.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
    cfg.INPUT.TRANSFORMS = ()
    cfg.INPUT.CROP_PADDING = 4
    
    # Optimizer settings
    cfg.OPTIM = CN()
    cfg.OPTIM.NAME = "sgd"
    cfg.OPTIM.LR = 0.002
    cfg.OPTIM.WEIGHT_DECAY = 5e-4
    cfg.OPTIM.MOMENTUM = 0.9
    cfg.OPTIM.SGD_DAMPNING = 0
    cfg.OPTIM.SGD_NESTEROV = False
    cfg.OPTIM.ADAM_BETA1 = 0.9
    cfg.OPTIM.ADAM_BETA2 = 0.999
    cfg.OPTIM.STAGED_LR = False
    cfg.OPTIM.NEW_LAYERS = ()
    cfg.OPTIM.BASE_LR_MULT = 0.1
    cfg.OPTIM.LR_SCHEDULER = "cosine"
    cfg.OPTIM.STEPSIZE = (-1, )
    cfg.OPTIM.GAMMA = 0.1
    cfg.OPTIM.MAX_EPOCH = 50
    cfg.OPTIM.WARMUP_EPOCH = 1
    cfg.OPTIM.WARMUP_TYPE = "constant"
    cfg.OPTIM.WARMUP_CONS_LR = 1e-5
    
    # Trainer settings
    cfg.TRAINER = CN()
    cfg.TRAINER.NAME = ""
    
    # MaPLe specific settings
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 16
    cfg.TRAINER.MAPLE.CTX_INIT = ""  # 添加缺失的配置项
    cfg.TRAINER.MAPLE.C_SCALE = 1.0
    cfg.TRAINER.MAPLE.C_INIT = "uniform"
    cfg.TRAINER.MAPLE.PREC = "fp16"
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # 可能也缺失
    
    # Dual Stream 配置
    cfg.TRAINER.MAPLE.DUAL_STREAM = CN()
    cfg.TRAINER.MAPLE.DUAL_STREAM.ENABLE = False
    cfg.TRAINER.MAPLE.DUAL_STREAM.ADAPTIVE_WEIGHT = False
    cfg.TRAINER.MAPLE.DUAL_STREAM.AUX_LOSS_WEIGHT = 0.3
    cfg.TRAINER.MAPLE.DUAL_STREAM.IMAGE_WEIGHT = 0.5
    
    # Fine-tune Text 配置
    cfg.TRAINER.MAPLE.FINETUNE_TEXT = CN()
    cfg.TRAINER.MAPLE.FINETUNE_TEXT.ENABLE = False
    cfg.TRAINER.MAPLE.FINETUNE_TEXT.LAYERS = 2
    cfg.TRAINER.MAPLE.FINETUNE_TEXT.LR_MULT = 0.1
    
    # Caption Loss 配置
    cfg.TRAINER.MAPLE.CAPTION_LOSS = CN()
    cfg.TRAINER.MAPLE.CAPTION_LOSS.ENABLE = False
    cfg.TRAINER.MAPLE.CAPTION_LOSS.WEIGHT = 0.25
    cfg.TRAINER.MAPLE.CAPTION_LOSS.TEMP = 2.5
    cfg.TRAINER.MAPLE.CAPTION_LOSS.RAMP_EPOCHS = 15
    
    # Caption Gating 配置
    cfg.TRAINER.MAPLE.CAPTION_LOSS.GATING = CN()
    cfg.TRAINER.MAPLE.CAPTION_LOSS.GATING.ENABLE = False
    cfg.TRAINER.MAPLE.CAPTION_LOSS.GATING.THRESH = 0.35
    
    # Staged Caption 配置
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED = CN()
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ENABLE = False
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ADAPTIVE = CN()
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ADAPTIVE.ENABLE = False
    cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.WEIGHTS = [0.333, 0.333, 0.334]
    
    # Training settings
    cfg.TRAIN = CN()
    cfg.TRAIN.PRINT_FREQ = 10
    cfg.TRAIN.CHECKPOINT_FREQ = 0
    
    # Test settings  
    cfg.TEST = CN()
    cfg.TEST.EVALUATOR = "Classification"
    cfg.TEST.PER_CLASS_RESULT = False
    cfg.TEST.COMPUTE_CMAT = False
    cfg.TEST.FINAL_MODEL = "best_val"
    
    # Other settings
    cfg.USE_CUDA = True
    cfg.SEED = 1
    cfg.OUTPUT_DIR = "./output"
    cfg.RESUME = ""
    cfg.VERBOSE = True
    
    return cfg


def extend_cfg_for_sentiment(cfg):
    """Extend config for sentiment analysis task."""
    from yacs.config import CfgNode as CN
    
    # Sentiment-specific settings
    cfg.SENTIMENT = CN()
    cfg.SENTIMENT.USE_TEXT = True  # Whether to use text modality
    cfg.SENTIMENT.USE_IMAGE = True  # Whether to use image modality
    cfg.SENTIMENT.FUSION_TYPE = "late"  # "early", "late", or "hybrid"
    cfg.SENTIMENT.TEXT_WEIGHT = 0.5  # Weight for text modality in fusion
    cfg.SENTIMENT.IMAGE_WEIGHT = 0.5  # Weight for image modality in fusion
    
    return cfg
