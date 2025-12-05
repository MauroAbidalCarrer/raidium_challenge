Documentation
    - felzenszwalb

ideas:
- find some way of loss normalization?
    - Use float/int8 convolution ? that would be dope
- make dice score run on pytorch?
- asynchronize train/evaluate?

todo:
- make training more data efficient, try to get the same (or a better) score with ~30 epochs
    - self supervised learning 
        - ssl pretraining:
            - Change output channels: 55 classes + 1 pixel value (maybe round to nearest power of two -> 64)
            - for each batch: add patches of noise to the input and a mask of where the noise was added
            - get model output, zero out image and model output where no noise was applied
            - express loss as MSE between whatever channel we chose as the pixle channel (maybe the last one?)
        - supervised learning finetuning: 
            - freeze the encoder?
            - train the model as usual on all but the pixel channel output
- hyper parameter tuning with optuna
- save checkpoint as artifact on wandb
- Fewer epochs per training runs 
- parrallel training run multiple processes on the same GPU?
- Get to top 1 with unet and (run training while eating)?
    - retrain on full set (train+valid), not just train set
    - use TTA
- self supervised learning:
    - patch prediction?
    - simCLR?
    - teacher/student? (probably later)
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
<!-- - speed up data aug by using torchvision.tranform.v2 -->
<!-- - use torch compile -->