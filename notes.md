Documentation
    - felzenszwalb

ideas:
Find close scans:
- find some way of loss normalization?
- use sigmoid activation for classes and set background only if all classes probs are less than 0.5.

todo:
- get to ~44 dice with just nnUNet:
    - data augmentation
    - use optuna
    - Speed up stuff
        - asynchronize train/evaluate
- understand why/fix the fact that higher batch sizes breaks training
- stratify train test split
- use lr scheduler?
- use TTA
- Use ensemble?

- make the repo veeeeeeeery clean, so I can flex it to everyone
    - create a second repo and delete this one?

<!-- - fix submission -->
<!-- - Use pixel wise class imbalance and weighting -->