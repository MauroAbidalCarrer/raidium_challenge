Documentation
    - felzenszwalb

ideas:
- find some way of loss normalization?
    - Use float/int8 convolution ? that would be dope

todo:
- Speed up stuff -> 13h
    - make dice score run on pytorch (use monai?)
    - understand why loss.item call is taking so much time and fix it
    - Leave x as uint8 and cast BATCH it to float 32 + apply normalization afterwards?
        - check this isn't what's breaking the training
    - use torch compile?
    - asynchronize train/evaluate and as many other things as possible
- save checkpoint as artifact on wandb asynchronously 13h30
- Get to top 1 with unet and (run training while eating)?
    - Even bigger model 
    - Even more data aug?
    - retrain on full set (train+valid), not just train set
    - use TTA
- self supervised learning:
    - patch prediction?
    - simCLR?
    - teacher/student? (probably later)
- hyper parameter tuning with optuna
- understand why/fix the fact that higher batch sizes breaks training
    - Could it be because we are using instance norm? maybe switch to batch norm?
- stratify train test split
- use lr scheduler?
- fine tune foundation model
- Use ensemble?

- make the repo veeeeeeeery clean, so I can flex it to everyone
    - create a second repo and delete this one?
    - Try to use monai loss/metrics when possible to reduce code base size.
    - Add readme
    - add prsentation noteook

<!-- - Use pixel wise class imbalance and weighting -->
<!-- - fix submission -->
<!-- - data augmentation
    - log data augmentation hp to wandb -->
<!-- - get to ~44 dice with just nnUNet(almost done):
    - increase model size (make sure receptive field reaches the entire picture)
    - use better data augmentation -->
