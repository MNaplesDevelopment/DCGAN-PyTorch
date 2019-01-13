# MNIST-DCGAN-PyTorch

Based of the tutorial PyTorch's website: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

I also took inspiration from fast ai: https://course.fast.ai/lessons/lesson12.html

I barrowed some code the PyTorch Website, I made minor changes to the generator and discriminator, the architectures are largely the same however I used more intuitive variable names. I also wrote my own training loop because the one in the tutorial was kind of hard to understand in my opinion. I also included the names of all the formal parameters so people unfamiliar with PyTorch should have an easy time understanding the code.

Generative Adverserial Networks are actually two networks in one, competing against eachother. The Discriminator a Convolutional Neural Network, whose job is to diffentiate between real and fake image, of in this case, images of digits. The Generator is Deconvolutional Neural Network, that will generate the actually images to feed to the discriminator in training (real images are also fed to discriminator at training time). It's crucial that these networks are mirror images of eachother.

Results after 1 epoch:

![GANoutput](/GANoutput.png)

I plan to train this for longer to get better result however I trained this on a CPU and 1 epoch took over 2 hours. The code can also be easily altered to take advantage of a GPU.
