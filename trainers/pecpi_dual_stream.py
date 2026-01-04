"""
MaPLe with Dual-Stream Fusion (No Concat Branch)

Key difference from three-stream:
- Only image stream and text stream (no concat fusion branch)
- Forces model to use class prompts for both modalities
- Prevents weight polarization to fusion branch
- More interpretable: weights show image vs text preference

Architecture:
  image_logits = image_features @ class_text_features.t()
  text_logits = sample_text_features @ class_text_features.t()
  final_logits = w_img * image_logits + w_text * text_logits
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from core.trainer_base import TrainerX, register_trainer
from core.optim import compute_accuracy, build_optimizer, build_lr_scheduler
from core.utils import load_pretrained_weights

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = n_ctx
        assert cfg_imsize == clip_imsize

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f"MaPLe design: Multi-modal Prompt Learning")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(ctx_dim, ctx_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(ctx_dim // 16, ctx_dim))
        ]))
        
        if cfg.TRAINER.MAPLE.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        
        if cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ADAPTIVE.ENABLE:
            self.staged_weight_logits = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer("staged_weights", torch.tensor(cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.WEIGHTS))

        self.proj = nn.Linear(ctx_dim, ctx_dim)
        if cfg.TRAINER.MAPLE.PREC == "fp16":
            self.proj.half()

        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH
        compound_prompts_text_tensor = torch.empty(self.compound_prompts_depth - 1, n_ctx, ctx_dim)
        nn.init.normal_(compound_prompts_text_tensor, std=0.02)
        self.compound_prompts_text = nn.Parameter(compound_prompts_text_tensor)

        visual_width = clip_model.visual.proj.shape[0] if hasattr(clip_model.visual, 'proj') else 768
        self.compound_prompt_projections = nn.ModuleList([
            nn.Linear(ctx_dim, visual_width)
            for _ in range(self.compound_prompts_depth - 1)
        ])

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))

        compound_prompts_text_list = [self.compound_prompts_text[i] for i in range(self.compound_prompts_text.shape[0])]
        return prompts, self.proj(self.ctx), compound_prompts_text_list, visual_deep_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_model_ref = clip_model
        
        self.dual_stream_enable = cfg.TRAINER.MAPLE.DUAL_STREAM.ENABLE
        self.dual_stream_adaptive = cfg.TRAINER.MAPLE.DUAL_STREAM.ADAPTIVE_WEIGHT
        self.dual_stream_aux_weight = cfg.TRAINER.MAPLE.DUAL_STREAM.AUX_LOSS_WEIGHT
        
        if self.dual_stream_enable:
            if self.dual_stream_adaptive:
                # Learnable weight for image vs text (only 2 streams)
                self.fusion_weight_logit = nn.Parameter(torch.tensor(0.0))  # Single logit
                print("Dual-stream adaptive fusion: learnable image/text balance")
            else:
                img_w = cfg.TRAINER.MAPLE.DUAL_STREAM.IMAGE_WEIGHT
                self.register_buffer("fusion_weight", torch.tensor(img_w))

    def _encode_text_base(self, text_tokens: torch.LongTensor) -> torch.Tensor:
        x = self.clip_model_ref.token_embedding(text_tokens).type(self.dtype)
        x = x + self.clip_model_ref.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        outputs = self.clip_model_ref.transformer([x, [], 0])
        x = outputs[0]
        x = x.permute(1, 0, 2)
        x = self.clip_model_ref.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model_ref.text_projection
        return x
    
    @torch.no_grad()
    def encode_captions_maple(self, caption_tokens: torch.LongTensor) -> torch.Tensor:
        return self._encode_text_base(caption_tokens)

    def forward(self, image, label=None, caption_tokens=None, staged_tokens=None, staged_weights=None, text_tokens=None, sample_text_tokens=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        class_text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        
        if hasattr(self.image_encoder, 'VPT_shallow'):
            image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        else:
            image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
        
        # Dual-stream fusion (image + text, NO concat branch)
        if self.dual_stream_enable and sample_text_tokens is not None:
            with torch.no_grad():
                sample_text_features = self._encode_text_base(sample_text_tokens.to(image_features.device))
                sample_text_features = sample_text_features / sample_text_features.norm(dim=-1, keepdim=True)
            
            # Two streams (equal footing)
            image_logits = logit_scale * image_features @ class_text_features.t()  # [batch, 3]
            text_logits = logit_scale * sample_text_features @ class_text_features.t()  # [batch, 3]
            
            # Adaptive or fixed fusion
            if self.dual_stream_adaptive:
                w_img = torch.sigmoid(self.fusion_weight_logit)  # ∈ [0,1]
                w_text = 1.0 - w_img
                logits = w_img * image_logits + w_text * text_logits
            else:
                w_img = self.fusion_weight
                w_text = 1.0 - w_img
                logits = w_img * image_logits + w_text * text_logits
        else:
            logits = logit_scale * image_features @ class_text_features.t()

        if self.prompt_learner.training:
            loss = F.cross_entropy(logits, label)
            
            # Auxiliary losses (encourage both streams to work)
            if self.dual_stream_enable and sample_text_tokens is not None and self.dual_stream_aux_weight > 0:
                aux_loss = F.cross_entropy(image_logits, label) + F.cross_entropy(text_logits, label)
                loss = loss + self.dual_stream_aux_weight * aux_loss
            
            # Caption contrastive loss
            if caption_tokens is not None:
                with torch.no_grad():
                    cap_feats = self.encode_captions_maple(caption_tokens.to(image_features.device))
                    cap_feats = cap_feats / cap_feats.norm(dim=-1, keepdim=True)
                
                cap_logit_scale = logit_scale / getattr(self, 'caption_temp', 1.0)
                cap_logits = cap_logit_scale * image_features @ cap_feats.t()
                cap_labels = torch.arange(cap_logits.shape[0], device=cap_logits.device)
                
                if hasattr(self, 'caption_gating') and self.caption_gating:
                    similarity = F.cosine_similarity(image_features, cap_feats, dim=-1)
                    gate_factor = (similarity > getattr(self, 'caption_gate_thresh', 0.2)).float().unsqueeze(1)
                    cap_logits = cap_logits * gate_factor
                
                cap_loss = F.cross_entropy(cap_logits, cap_labels)
                
                ramp_epochs = getattr(self, 'caption_ramp_epochs', 0)
                ramp_factor = 1.0
                if ramp_epochs and hasattr(self, 'current_epoch'):
                    ramp_factor = min(1.0, (self.current_epoch + 1) / float(max(1, ramp_epochs)))
                
                caption_weight = getattr(self, 'caption_loss_weight', 0.5)
                loss = loss + caption_weight * ramp_factor * cap_loss
            
            # Three-stage caption supervision
            if staged_tokens is not None:
                if staged_weights is None:
                    sw = torch.softmax(self.prompt_learner.staged_weight_logits, dim=0)
                else:
                    sw = staged_weights
                
                for idx, tokens in enumerate(staged_tokens):
                    if tokens is None or sw[idx] <= 0:
                        continue
                    with torch.no_grad():
                        s_feats = self.encode_captions_maple(tokens.to(image_features.device))
                        s_feats = s_feats / s_feats.norm(dim=-1, keepdim=True)
                    
                    s_logit_scale = logit_scale / getattr(self, 'caption_temp', 1.0)
                    s_logits = s_logit_scale * image_features @ s_feats.t()
                    s_labels = torch.arange(s_logits.shape[0], device=s_logits.device)
                    
                    if hasattr(self, 'caption_gating') and self.caption_gating:
                        similarity = F.cosine_similarity(image_features, s_feats, dim=-1)
                        gate_factor = (similarity > getattr(self, 'caption_gate_thresh', 0.2)).float().unsqueeze(1)
                        s_logits = s_logits * gate_factor
                    
                    s_loss = F.cross_entropy(s_logits, s_labels)
                    
                    if ramp_epochs and hasattr(self, 'current_epoch'):
                        ramp_factor = min(1.0, (self.current_epoch + 1) / float(max(1, ramp_epochs)))
                        loss = loss + sw[idx] * ramp_factor * s_loss
                    else:
                        loss = loss + sw[idx] * s_loss
            
            return loss
        
        return logits


@register_trainer("MaPLe")
class MaPLe(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP (Dual-Stream: image+text, no concat branch)")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        self.model.caption_loss_weight = cfg.TRAINER.MAPLE.CAPTION_LOSS.WEIGHT if cfg.TRAINER.MAPLE.CAPTION_LOSS.ENABLE else 0.0
        self.model.caption_temp = cfg.TRAINER.MAPLE.CAPTION_LOSS.TEMP
        self.model.caption_ramp_epochs = cfg.TRAINER.MAPLE.CAPTION_LOSS.RAMP_EPOCHS
        self.model.caption_gating = cfg.TRAINER.MAPLE.CAPTION_LOSS.GATING.ENABLE
        self.model.caption_gate_thresh = cfg.TRAINER.MAPLE.CAPTION_LOSS.GATING.THRESH
        
        self.staged_enable = cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ENABLE and cfg.TRAINER.MAPLE.CAPTION_LOSS.ENABLE
        self.staged_weights = cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.WEIGHTS
        self.staged_adaptive = cfg.TRAINER.MAPLE.CAPTION_LOSS.STAGED.ADAPTIVE.ENABLE and self.staged_enable

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        
        # Enable fusion_weight_logit if adaptive
        if cfg.TRAINER.MAPLE.DUAL_STREAM.ENABLE and cfg.TRAINER.MAPLE.DUAL_STREAM.ADAPTIVE_WEIGHT:
            self.model.fusion_weight_logit.requires_grad_(True)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {len(enabled)} parameters")
        print(f"First few: {list(enabled)[:5]}")

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, extra = self.parse_batch_train(batch)
        caption_tokens = None
        staged_tokens = None
        staged_weights = None
        sample_text_tokens = None
        
        if isinstance(extra, dict):
            caption_tokens = extra.get('caption_tokens', None)
            staged_tokens = extra.get('staged_tokens', None)
            sample_text_tokens = extra.get('sample_text_tokens', None)
            if not self.staged_adaptive:
                staged_weights = extra.get('staged_weights', None)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        if hasattr(model, 'module'):
            model.module.current_epoch = self.epoch
        else:
            model.current_epoch = self.epoch

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label, caption_tokens, staged_tokens, staged_weights, None, sample_text_tokens)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label, caption_tokens, staged_tokens, staged_weights, None, sample_text_tokens)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item(), "acc": 0.0}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        batch_size = input.shape[0]
        
        tone = batch.get('caption_tone', [""] * batch_size)
        content = batch.get('caption_content', [""] * batch_size)
        emotion = batch.get('caption_emotion', [""] * batch_size)
        raw_text = batch.get('text', [""] * batch_size)
        
        if not self.model.training:
            if self.cfg.TRAINER.MAPLE.DUAL_STREAM.ENABLE:
                try:
                    sample_text_tokens = clip.tokenize(raw_text, truncate=True)
                except Exception:
                    sample_text_tokens = clip.tokenize([""] * batch_size, truncate=True)
                return input, label, sample_text_tokens
            else:
                return input, label
        
        caption_tokens = None
        staged_tokens = []
        sample_text_tokens = None
        
        if self.cfg.TRAINER.MAPLE.CAPTION_LOSS.ENABLE:
            texts = []
            for i in range(batch_size):
                full_caption = f"{raw_text[i]}. {tone[i]}. {content[i]}. {emotion[i]}".strip()
                texts.append(full_caption)
            
            try:
                caption_tokens = clip.tokenize(texts, truncate=self.cfg.TRAINER.MAPLE.CAPTION_LOSS.TRUNCATE)
            except Exception:
                caption_tokens = clip.tokenize([""] * batch_size, truncate=True)
            caption_tokens = caption_tokens.to(self.device)
            
            if self.staged_enable:
                staged_texts = [
                    [tone[i] for i in range(batch_size)],
                    [content[i] for i in range(batch_size)],
                    [emotion[i] for i in range(batch_size)]
                ]
                for s_text_list in staged_texts:
                    try:
                        tokens = clip.tokenize(s_text_list, truncate=self.cfg.TRAINER.MAPLE.CAPTION_LOSS.TRUNCATE)
                    except Exception:
                        tokens = clip.tokenize([""] * batch_size, truncate=True)
                    staged_tokens.append(tokens.to(self.device))
            else:
                staged_tokens = None
        
        if self.cfg.TRAINER.MAPLE.DUAL_STREAM.ENABLE:
            try:
                sample_text_tokens = clip.tokenize(raw_text, truncate=True)
            except Exception:
                sample_text_tokens = clip.tokenize([""] * batch_size, truncate=True)
            sample_text_tokens = sample_text_tokens.to(self.device)
        
        extra = {
            "caption_tokens": caption_tokens,
            "staged_tokens": staged_tokens,
            "sample_text_tokens": sample_text_tokens,
            "staged_weights": (torch.tensor(self.staged_weights, device=self.device) if (self.staged_enable and not self.staged_adaptive) else None)
        }
        return input, label, extra

    def model_inference(self, input, sample_text_tokens=None):
        return self.model(input, sample_text_tokens=sample_text_tokens)
    
    def after_epoch(self):
        super().after_epoch()
        
        # Print dual-stream weight
        if self.cfg.TRAINER.MAPLE.DUAL_STREAM.ENABLE and self.cfg.TRAINER.MAPLE.DUAL_STREAM.ADAPTIVE_WEIGHT:
            model_unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_unwrapped, 'fusion_weight_logit'):
                w_img = torch.sigmoid(model_unwrapped.fusion_weight_logit).item()
                w_text = 1.0 - w_img
                print(f"Dual-stream weights @epoch{self.epoch+1}: img={w_img:.3f}, text={w_text:.3f}")





