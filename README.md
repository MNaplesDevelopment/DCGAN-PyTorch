# MNIST-DCGAN-PyTorch

Generative Adverserial Networks are actually two networks in one, competing against eachother. The Discriminator a Convolutional Neural Network, whose job is to diffentiate between real and fake images. The Generator is Deconvolutional Neural Network, that will generate the actual images to feed to the discriminator in training (real images are also fed to discriminator at training time). It's crucial that these networks are mirror images of eachother.

# Results CelebA
![Celeb](/imgs/Celeb.png)


Results after 2 epoch:

![GAN-Output](/imgs/GAN-Output.png)

And here's a look inside the Generator! Each picture contains each channel at a layer. The images start at 4x4 pixels and double in size until they are 64x64.

![GAN-Deconv1](/imgs/GAN-Deconv1.png)

![GAN-Deconv2](/imgs/GAN-Deconv2.png)

![GAN-Deconv3](/imgs/GAN-Deconv3.png)

![GAN-Deconv4](/imgs/GAN-Deconv4.png)

![GAN-Deconv5](/imgs/GAN-Deconv5.png)

# More Deep Learning

More to come soon! I'm currently working on progressively growing GAN's as described in this paper from NVIDIA: https://arxiv.org/abs/1710.10196

Also check out my school project which takes a very Natural Language Processing approach to bug detection: Deep Learning to Decet Logical Bugs in Software: https://github.com/TeamLigers/bug-prediction/blob/master/src/RNN_model.ipynb
