# Experiment_EaPU
The simulation code of EaPU. 

The code includes simulation experiments of the resnet with cifar10/cifar100, the swin transformer with mini-imagenet and the srresnet with ImageNet.

# Pretrained weight
The pretrained weight of the swin transformer is available at https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth.

# Data download
Training data in SRResNet download at https://huggingface.co/datasets/goodfellowliu/SRGAN_ImageNet/resolve/main/SRGAN_ImageNet.zip;  
Testing data in SRResNet download at https://huggingface.co/datasets/goodfellowliu/Set5/resolve/main/Set5.zip.    
Then, we can unzip the package and place the folder in `train/train_srresnet/data`.

Mini_imagenet data can be available at https://drive.google.com/file/d/1rK4ihgKpW2iIIs5yWnSFyFYa4FURCxM9/view?usp=drive_link.  
Then, we can uncompress the package and place the folder `/images` in `train/train_mini_imagenet/mini_imagenet`.

# Acknowledgement
https://github.com/kuangliu/pytorch-cifar

https://github.com/Lornatang/SRGAN-PyTorch

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

