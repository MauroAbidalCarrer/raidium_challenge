# ENS and Raidium data challenge <img src="./readme_ressources/raidum_logo.png" width="100" height="100"> <img src="./readme_ressources/ENS_Logo_TL.jpg" width="100" height="100">
This is the repo that got me 6th place in 15th of december 2025 with a dice score of 0,52.  
15th of december 2025 was supposed to mark the end of the competition but it got postponed.  
In the mean time, other participants have made it above my score 😥 and I am now 10th.  
But I might give it another shot and try to increase my score to get first place.  

### Problem:
We are give a dataset of 2000 CT-Scam images, ~800 of which are partially segmented.  
Our goal will be to train a segementation model to segment the rest of the images.  
No external medical imaging datasets and models are allowed.  
However generic datasets and models are allowed.  
Bellow is an illustration of the task, the left image is a the image we want to segment and on the right are the colorised body parts the must segment.  
![task_illustration](./readme_ressources/task_illustration.png)
> Note: Here it seems like all the body parts are segmented.  
> But the labeled images we are provided with are not fully segmented.  

### My solutions:
My best performing solution is a UNet trained with:
-   A batch size of 1
    > For some unkown reason the model was very sensitive to the batch size.  
    > I assume this is because the labeled images are partially segmented and the classes distribution is very uneven.  
-   Heavy data augmentation including a lot of random erasing
-   AdamW optimizer
Here is a small video of the model training (not the model I used to rank 6th):
![model_training](https://youtu.be/csb_RyoLXdA)

This first supervised solution does not exploit all the data available since it only uses the 800 partially labeled images.
I actually spent most of the competition on semi-supervised learning solutions.
Sadly, I couldn't get them to outperform the supervised one.

The most promessing semi-supervised one is the SwinViT pretrained with simMIM (simple Masked Image Modeling) and finetuned in the same way as the UNet was trained.
You can view some of the reconstructed images in the viz_hf_swin_reoncstruction notebook.