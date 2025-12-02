Documentation
    - felzenszwalb

ideas:
Find close scans:
- find some way of loss normalization?
- use sigmoid activation for classes and set background only if all classes probs are less than 0.5.
- Use trained model to segment unlabeled and use only very confident guesses

todo:
- get to ~44 dice with just nnUNet:
    - increase model size (make sure receptive field reaches the entire picture)
    - use optuna
    - Speed up stuff
        - Leave x as uint8 and cast BATCH it to float 32 + apply normalization afterwards?
        - asynchronize train/evaluate
- understand why/fix the fact that higher batch sizes breaks training
    - Could it be because we are using instance norm? maybe switch to batch norm?
- stratify train test split
- use lr scheduler?
- use TTA
- Use ensemble?

- make the repo veeeeeeeery clean, so I can flex it to everyone
    - create a second repo and delete this one?

<!-- - Use pixel wise class imbalance and weighting -->
<!-- - fix submission -->
<!-- - data augmentation
    - log data augmentation hp to wandb -->