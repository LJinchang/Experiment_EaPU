# Experiment_EaPU
The simulation code of EaPU. 

The code includes simulation experiments of the ResNet with CIFAR10/CIFAR100, the Swin Transformer with mini-imagenet, and the SRResNet with ImageNet.

# Environment
Relevant simulation experiments were conducted using Python version 3.8.20, and all other higher Python versions are compatible with the code.

For the CUDA version of PyTorch, the following command can be used for installation:
  ```shell
  pip3 install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
  ```
# Pretrained weight
The pretrained weight of the Swin Transformer is available at https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth.

Then, we can place the pretrained weight `swin_tiny_patch4_window7_224.pth` in `train/train_mini_imagenet`.

# Data download
Training data in SRResNet download at https://huggingface.co/datasets/goodfellowliu/SRGAN_ImageNet/resolve/main/SRGAN_ImageNet.zip;  
Testing data in SRResNet download at https://huggingface.co/datasets/goodfellowliu/Set5/resolve/main/Set5.zip.    
Then, we can extract data from the archive and place the folder `ImageNet` and `Set5` in `train/train_srresnet/data`.

Mini_imagenet data can be available at https://drive.google.com/file/d/1rK4ihgKpW2iIIs5yWnSFyFYa4FURCxM9/view?usp=drive_link.  
Then, we can uncompress the archive and place the folder `images` in `train/train_mini_imagenet/mini_imagenet`.

# Experiment result
## ResNet34 with cifar10
![comparison](train/train_cifar/results/comparison.png)

The figure shows the comparison of training with EaPU and the original algorithm. AdamEaPU is the suggested update method, while AdamNoise represents the original method. The postfix in the legend (a-b-c-d) represents the learning rate, the clipped value (clipped value = ΔW<sub>th</sub>/R<sub>wg</sub>), the standard deviation of the writing noise, and the R<sub>wg</sub>, respectively. When the noise is 0, the AdamNoise is the ideal learning process. Training with EaPU achieves much better results than the original method (89.71 % vs 17.55 %, writing/update noise of 2 μS).

Detailed results and analysis of other simulations can be found in the article "Error-aware Probabilistic Training for Memristive Neural Networks".

# Acknowledgement
https://github.com/kuangliu/pytorch-cifar

https://github.com/Lornatang/SRGAN-PyTorch

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

# License

This project is covered under the **Apache 2.0 License**.

