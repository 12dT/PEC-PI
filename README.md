# PEC-PI: Progressive Emotion Construction via Prompt Interaction for Multimodal Sentiment Analysis

Official implementation of **PEC-PI**, a multi-stage framework for multimodal sentiment analysis that progressively constructs affective representations through **Tone**, **Content**, and **Emotion**.

## Introduction

Multimodal sentiment analysis requires joint reasoning over textual and visual cues. However, sentiment is often implicit and gradually emerges from perceptual observations, semantic understanding, and affective interpretation, rather than being directly captured through one-step multimodal fusion.

PEC-PI addresses this challenge by modeling multimodal emotion construction as a three-stage progressive process:

- **Tone**: captures perceptual cues, such as color, illumination, and overall atmosphere
- **Content**: identifies semantic entities, scenes, actions, and interactions
- **Emotion**: derives high-level affective interpretations from perceptual and semantic evidence

To support this progressive emotion construction process, PEC-PI introduces:

1. **Multi-stage semantic guidance** to provide stage-specific supervision from Tone to Content and Emotion
2. **Prototype-centered prompt interaction** to facilitate cross-modal affective information transfer
3. **Dual-path contrastive alignment** to learn structured and sentiment-discriminative multimodal representations

## Framework Overview

<p align="center">
  <img src="figures/image.png" width="85%">
</p>

PEC-PI progressively constructs multimodal affective representations from **Tone** to **Content** and finally **Emotion**. Stage-wise semantic guidance supports this evolving process, while prototype-centered prompt interaction and dual-path contrastive alignment promote effective cross-modal interaction and structured affective representation learning.

The source code will be released publicly. This repository currently serves as a placeholder for reference and linkage purposes.

Thank you for your interest.
