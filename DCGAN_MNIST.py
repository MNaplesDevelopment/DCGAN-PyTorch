import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import time
import imageio
from tkinter import *
from PIL import Image, ImageTk
import random

num_channels = 3
# Reshape data to 64x64 and normalize the values so they are between -1 and 1.
trans = transforms.Compose([transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

celeba_dataset = datasets.ImageFolder(root='C:/celeba', transform=trans)
train_loader = torch.utils.data.DataLoader(dataset=celeba_dataset, batch_size=32, shuffle=False)


class DCGAN_D(nn.Module):
    """
    Discriminator Network. (Convolutional)
    """
    def __init__(self, num_filters):
        super(DCGAN_D, self).__init__()
        # input: 128x128 - number of channels(1) for grayscale
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        # input: 64x64 - num_filters
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=num_filters*2)
        # input: 32x32 - num_filters * 2
        self.conv3 = nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=num_filters*4)
        # input: 16x16 - num_filters * 4
        self.conv4 = nn.Conv2d(in_channels=num_filters*4, out_channels=num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=num_filters*8)
        # input: 8x8 - num_filters * 8
        self.conv5 = nn.Conv2d(in_channels=num_filters*8, out_channels=num_filters*16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=num_filters*16)
        # input: 4x4 - num_filters * 16
        self.conv6 = nn.Conv2d(in_channels=num_filters*16, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.sigmoid(self.conv6(x))
        return x


class DCGAN_G(nn.Module):
    """
    Generator Network. (Deconvolutional)
    """
    def __init__(self, latent_size, num_filters):
        super(DCGAN_G, self).__init__()
        # input: 4x4 - num_filters * 16
        self.deconv1 = nn.ConvTranspose2d(latent_size, out_channels=num_filters*16, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=num_filters*16)
        # input: 8x8 - num_filters * 8
        self.deconv2 = nn.ConvTranspose2d(in_channels=num_filters*16, out_channels=num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=num_filters*8)
        # input: 16x16 - num_filters * 4
        self.deconv3 = nn.ConvTranspose2d(in_channels=num_filters*8, out_channels=num_filters*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=num_filters*4)
        # input: 32x32 - num_filters * 2
        self.deconv4 = nn.ConvTranspose2d(in_channels=num_filters*4, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=num_filters*2)
        # input: 64x64 - num_filters
        self.deconv5 = nn.ConvTranspose2d(in_channels=num_filters*2, out_channels=num_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features=num_filters)
        # input: 128x128 - number of channels (1) for grayscale
        self.deconv6 = nn.ConvTranspose2d(in_channels=num_filters, out_channels=num_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, input):
        out = []
        x = F.relu(self.bn1(self.deconv1(input)))
        out.append(x)
        x = F.relu(self.bn2(self.deconv2(x)))
        out.append(x)
        x = F.relu(self.bn3(self.deconv3(x)))
        out.append(x)
        x = F.relu(self.bn4(self.deconv4(x)))
        out.append(x)
        x = F.relu(self.bn5(self.deconv5(x)))
        x = F.tanh(self.deconv6(x))
        out.append(x)

        return out, x

    def load(self):
        self.load_state_dict(torch.load('Networks/DCGANG_128xFaces5.pt'))


filters, feature_space = 32, 128

netG = DCGAN_G(latent_size=feature_space, num_filters=filters)
netD = DCGAN_D(num_filters=filters)


def create_noise(b):
    """
    Create an image of random noise.
    :param b - batch size
    """
    return torch.zeros(b, feature_space, 1, 1).normal_(0, 1)


# Binary Cross Entropy loss
criterion = nn.BCELoss()

# Optimizers
optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4)
optimizerG = optim.RMSprop(netG.parameters(), lr=1e-4)


def train(epochs):
    """
    Main training loop.
    """
    for epoch in range(epochs):
        x = 0
        for batch, labels in train_loader:
            # batch size: [32, 1, 64, 64]
            netD.zero_grad()

            batch_size = batch.shape[0]
            y_real = torch.ones(batch_size)     # labels for the networks, the discriminator is training to
            y_fake = torch.zeros(batch_size)    # differentiate between real and fake images, so we label them accordingly

            batch, y_real, y_fake = Variable(batch), Variable(y_real), Variable(y_fake)
            D_output_real = netD(batch).squeeze()    # predictions of the 32 real images.
            D_real_loss = criterion(D_output_real, y_real)  # Binary Cross Entropy loss

            noise = create_noise(batch_size)
            G_output = netG(noise)

            D_output_fake = netD(G_output).squeeze()    # prediction of the 32 generated images
            D_fake_loss = criterion(D_output_fake, y_fake)  # Binary Cross Entropy loss
            D_fake_score = D_output_fake.data.mean()

            D_train_loss = D_fake_loss + D_real_loss
            D_train_loss.backward()
            optimizerD.step()

            # Generator Training
            netG.zero_grad()

            noise = create_noise(batch_size)
            G_output = netG(noise)
            D_output = netD(G_output).squeeze()
            G_train_loss = criterion(D_output, y_real)
            G_train_loss.backward()
            optimizerG.step()

            if x % 50 == 0:
                print('D loss: {}\tG loss: {}\tProgress: {}/{}'.format(D_train_loss, G_train_loss, x, 1875))

            x += 1


# train(1)
#
# torch.save(netG.state_dict(), "DCGAN1.pt")

netG.load_state_dict(torch.load('Networks/DCGANG_128xFaces5.pt'))

# _, img = netG.forward(create_noise(20))
# img = img.detach().cpu().numpy()
#
# fig = plt.figure(figsize=(8, 8))
# for i in range(20):
#     fig.add_subplot(5, 4, i + 1)
#     plt.imshow(np.transpose(img[i], (1, 2, 0)))
# plt.show()


def plot_layers(greyscale):
    out, img = netG(create_noise(1))
    for i in range(len(out)):
        out[i] = out[i].detach().cpu().numpy()

    # 512
    fig = plt.figure(figsize=(8, 8))
    for i in range(100):
        img = out[0][0, i, :, :]
        fig.add_subplot(10, 10, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # 256
    fig = plt.figure(figsize=(8, 8))
    for i in range(100):
        img = out[1][0, i, :, :]
        fig.add_subplot(10, 10, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # 128
    fig = plt.figure(figsize=(8, 8))
    for i in range(100):
        img = out[2][0, i, :, :]
        fig.add_subplot(10, 10, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # 64
    fig = plt.figure(figsize=(8, 8))
    for i in range(64):
        img = out[3][0, i, :, :]
        fig.add_subplot(8, 8, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # 64
    if greyscale:
        fig = plt.figure(figsize=(8, 8))
        img = out[4][0, 0, :, :]
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        #fig = plt.figure(figsize=(8, 8))
        img = out[4][0, :, :, :].transpose(1, 2, 0)
        img = ((img - img.min()) * (1 / img.max() - img.min()) * 50).astype('uint8')
        plt.imshow(img)
        plt.show()


def plot_images(grayscale):
    fig = plt.figure(figsize=(8, 8))
    _, img = netG(create_noise(20))
    if grayscale:
        for i in range(20):
            arr = img.detach().cpu().numpy()
            fig.add_subplot(5, 4, i + 1)
            plt.imshow(arr[i][0], cmap='gray')
    else:
        img = img.detach().cpu().numpy()
        arr = ((img - img.min()) * (1 / img.max() - img.min()) * 70).astype('uint8')
        for i in range(20):
            fig.add_subplot(5, 4, i + 1)
            plt.imshow(arr[i].transpose(1, 2, 0))

    plt.show()


#plot_images(False)
#plot_layers(False)


def Board():
    _, img = netG.forward(create_noise(1))
    img = img.detach().cpu().numpy()[0][0]
    img = ((img - img.min()) * (1/img.max() - img.min()) * 255)

    def create_vector():
        arr = np.empty((1, 128, 1, 1))
        for i in range(128):
            arr[0][i][0][0] = sliders[i].get()
        ten = torch.tensor(arr)
        return ten.float()

    def update_picture(x):
        _, img = netG.forward(create_vector())
        img = img.detach().cpu().numpy()[0]
        img = ((img - img.min()) * (1 / img.max() - img.min()) * 50).astype('uint8')
        img = img.transpose((1, 2, 0))
        photo = ImageTk.PhotoImage(image=Image.fromarray(img, 'RGB').resize((256, 256), Image.ANTIALIAS))

        label.configure(image=photo)
        label.image = photo

    def interpolate():
        ndx = random.randint(0, 123)
        direction = []
        arr = [-1, 1]
        for i in range(5):
            direction.append(arr[random.randint(0, 1)])
        for i in range(10):
            for x in range(5):
                sliders[ndx].set(sliders[ndx].get() + (direction[x] * 0.03))
                ndx += 1
            update_picture(1)
            ndx -= 5
            time.sleep(0.05)

    def randomize():
        for x in sliders:
            x.set(random.uniform(-1, 1))

    def reset():
        for x in sliders:
            x.set(0)

    tk = Tk()

    photo = ImageTk.PhotoImage(image=Image.fromarray(img))
    label = Label(image=photo, height=256, width=256)
    label.grid(row=0, column=0)

    sliders = []
    for i in range(128):
        sliders.append(Scale(tk, from_=1, to=-1, resolution=0.01, command=update_picture))
        sliders[i].grid(row=int(i/32), column=i % 32 + 2)

    create = Button(text="interpolate", command=interpolate).grid(row=1, column=0)
    rand = Button(text='randomize', command=randomize).grid(row=2, column=0)
    zero = Button(text='reset', command=reset).grid(row=3, column=0)

    tk.mainloop()


#Board()


def interpolate(speed):
    vec = create_noise(1)[0, :, 0, 0].numpy()
    increments = []
    min, max = np.min(vec), np.max(vec)
    images = []
    for i in range(len(vec)):
        if vec[i] > 0.5:
            increments.append((min + vec[i]) / speed)
        elif vec[i] < -0.5:
            increments.append((max - vec[i]) / speed)
        else:
            direction = random.randint(0, 1)
            if direction == 1:
                increments.append((max - vec[i]) / speed)
            else:
                increments.append((min + vec[i]) / speed)
    increments = np.asarray(increments) * 0.9
    for i in range(speed * 4):
        if i < 10:
            vec[:32] += increments[:32]
            vec[65:96] -= increments[65:96]
        elif i < 20:
            vec[33:64] += increments[33:64]
            vec[65:96] += increments[65:96]
        elif i < 30:
            vec[65:96] += increments[65:96]
        else:
            vec[97:] += increments[97:]
            vec[33:64] -= increments[33:64]
        vec.dtype = np.float32
        tensor = torch.tensor(vec)
        tensor = tensor.view(1, -1, 1, 1)
        _, img = netG.forward(tensor)
        img = img.detach().cpu().numpy()[0]
        img = ((img - img.min()) * (1 / img.max() - img.min()) * 80).astype('uint8')
        img = img.transpose((1, 2, 0))
        photo = Image.fromarray(img, 'RGB').resize((256, 256), Image.ANTIALIAS)
        img = np.array(photo)
        images.append(img)
    for i in reversed(images):
        images.append(i)
    images = np.asarray(images)
    imageio.mimsave('test.gif', images)


#interpolate(10)
