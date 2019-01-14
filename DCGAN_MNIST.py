import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle


batch_size = 32

# Reshape data to 64x64 and normalize the values so they are between 0 and 1.
trans = transforms.Compose([transforms.Resize(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Set download to true if first time downloading
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=trans)
train_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=batch_size, shuffle=True)


class DCGAN_D(nn.Module):
    """
    Discriminator Network. (Convolutional)
    """
    def __init__(self, num_filters):
        super(DCGAN_D, self).__init__()
        # input: 64x64 - number of channels(1) for grayscale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        # input: 32x32 - num_filters
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=num_filters*2)
        # input: 16x16 - num_filters * 2
        self.conv3 = nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=num_filters*4)
        # input: 8x8 - num_filters * 4
        self.conv4 = nn.Conv2d(in_channels=num_filters*4, out_channels=num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=num_filters*8)
        # input: 4x4 - num_filters * 8
        self.conv5 = nn.Conv2d(in_channels=num_filters*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.conv1(input)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))
        return x


class DCGAN_G(nn.Module):
    """
    Generator Network. (Deconvolutional)
    """
    def __init__(self, latent_size, num_filters):
        super(DCGAN_G, self).__init__()
        # output: 4x4 - num_filters * 8
        self.deconv1 = nn.ConvTranspose2d(latent_size, out_channels=num_filters*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=num_filters*8)
        # output: 8x8 - num_filters * 4
        self.deconv2 = nn.ConvTranspose2d(in_channels=num_filters*8, out_channels=num_filters*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=num_filters*4)
        # output: 16x16 - num_filters * 2
        self.deconv3 = nn.ConvTranspose2d(in_channels=num_filters*4, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=num_filters*2)
        # output: 32x32 - num_filters
        self.deconv4 = nn.ConvTranspose2d(in_channels=num_filters*2, out_channels=num_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=num_filters)
        # output: 64x64 - number of channels (1) for grayscale
        self.deconv5 = nn.ConvTranspose2d(in_channels=num_filters, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, input):
        x = F.relu(self.bn1(self.deconv1(input)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x


filters, feature_space = 64, 128

netG = DCGAN_G(latent_size=feature_space, num_filters=filters).cuda()
netD = DCGAN_D(num_filters=filters).cuda()


def create_noise(b):
    """
    Create vector of random noise.
    :param b - batch size
    """
    return Variable(torch.zeros(b, feature_space, 1, 1).normal_(0, 1).cuda())


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
            y_fake = torch.zeros(batch_size)    # differentiate between real and fake images, so we label the accordingly

            batch, y_real, y_fake = Variable(batch.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
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
                print('D loss: {}\tG loss: {}\tProgress: {}/{}'.format(D_train_loss, G_train_loss, x, 60000/batch_size))

            x += 1


train(2)

torch.save(netG.state_dict(), "DCGAN-G.pt")
torch.save(netD.state_dict(), "DCGAN-D.pt")




