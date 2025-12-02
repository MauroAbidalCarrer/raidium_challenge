Documentation
    - felzenszwalb

ideas:
- find some way of loss normalization?

todo:
- get to ~44 dice with just nnUNet(almost done):
    - increase model size (make sure receptive field reaches the entire picture)
    - use better data augmentation
- self super vised learning:
    - patch prediction?
- use optuna
- implement checkpointing
    - save checkpoints to wandb
- Speed up stuff
    - Leave x as uint8 and cast BATCH it to float 32 + apply normalization afterwards?
    - asynchronize train/evaluate and as many other things as possible
    - makde dice score run on pytorch (use monai?)
    - Use float/int8 convolution ? that would be dope
- understand why/fix the fact that higher batch sizes breaks training
    - Could it be because we are using instance norm? maybe switch to batch norm?
- stratify train test split
- use lr scheduler?
- use TTA
- Use ensemble?

- make the repo veeeeeeeery clean, so I can flex it to everyone
    - create a second repo and delete this one?
    - Try to use monai loss/metrics when possible.
    - Add readme
    - add prsentation noteook

<!-- - Use pixel wise class imbalance and weighting -->
<!-- - fix submission -->
<!-- - data augmentation
    - log data augmentation hp to wandb -->
