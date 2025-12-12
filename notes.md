Documentation
    - felzenszwalb

ideas:
- optimization:
    - Use float/int8 convolution ? that would be dope
    - make dice score run on pytorch?
- model performance:
    - find some way of loss normalization?
    - Use swinVIT
    - Use float32 all the way
    - use blur and gaussian noise for data aug
- workflow:
    - save checkpoint as artifact on wandb

todo:
- fix training:
    - make sure we can restart training from checkpoint(kinda done)
    - understand why we get a better validation dice score when restarting training from a checkpoint
    - ideally save checkpoint on wandb
- in parallel:
    - Get better score with unet:
        - add gaussian blur
        - decrease test size
        - train for longer from checkpoint
    - TTA
    - ViT pretraining with downscaled image
at the end of the competion:

submissions:
- today:
    - high score unet
    - unet with blur+balanced sampling
    - unet TTA
- tomorrow:
    - downscaled vit?
    - unet or vit TTA

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
<!-- - speed up data aug by using torchvision.tranform.v2 -->
<!-- - use torch compile -->
<!-- - save checkpoint  -->