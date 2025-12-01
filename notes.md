Documentation
    - felzenszwalb

ideas:
Find close scans:
- find some way of loss normalization?
- use sigmoid activation for classes and set background only if all classes probs are less than 0.5.

todo:
- fix training:
    - understand why/fix the fact that higher batch sizes breaks training
    - data augmentation
    - adversarial training
- fix submission
- Add images to wandb logs.
- use optuna
- stratify train test split
- Use augmentations to avoid overfittings
- use lr scheduler?
- use TTA

<!-- - Use pixel wise class imbalance and weighting -->