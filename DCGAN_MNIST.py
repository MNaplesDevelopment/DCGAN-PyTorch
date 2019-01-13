import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Reshape data to 64x64 and normalize the values so they are between -1 and 1.
trans = transforms.Compose([transforms.Resize(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Set download to true if first time downloading
mnist_data = datasets.MNIST(root='./data', train=True, download=False, transform=trans)
train_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=32, shuffle=True)


class DCGAN_D(nn.Module):
    """
    Discriminator Network. (Convolutional)
    """
    def __init__(self, num_filters):
        super(DCGAN_D, self).__init__()
        self.main = nn.Sequential(
            # input: 64x64 - number of channels(1) for grayscale
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_filters),
            nn.LeakyReLU(0.2, inplace=True),
            # input: 32x32 - num_filters
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_filters*2),
            nn.LeakyReLU(0.2, inplace=True),
            # input: 16x16 - num_filters * 2
            nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_filters*4),
            nn.LeakyReLU(0.2, inplace=True),
            # input: 8x8 - num_filters * 4
            nn.Conv2d(in_channels=num_filters*4, out_channels=num_filters*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_filters*8),
            nn.LeakyReLU(0.2, inplace=True),
            # input: 4x4 - num_filters * 8
            nn.Conv2d(in_channels=num_filters*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class DCGAN_G(nn.Module):
    """
    Generator Network. (Deconvolutional)
    """
    def __init__(self, latent_size, num_filters):
        super(DCGAN_G, self).__init__()
        self.main = nn.Sequential(
            # input: 4x4 - num_filters * 8
            nn.ConvTranspose2d(latent_size, out_channels=num_filters*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=num_filters*8),
            nn.ReLU(True),
            # input: 8x8 - num_filters * 4
            nn.ConvTranspose2d(in_channels=num_filters*8, out_channels=num_filters*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_filters*4),
            nn.ReLU(True),
            # input: 16x16 - num_filters * 2
            nn.ConvTranspose2d(in_channels=num_filters*4, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_filters*2),
            nn.ReLU(True),
            # input: 32x32 - num_filters
            nn.ConvTranspose2d(in_channels=num_filters*2, out_channels=num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(True),
            # input: 64x64 - number of channels (1) for grayscale
            nn.ConvTranspose2d(in_channels=num_filters, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


filters, feature_space = 32, 100

netG = DCGAN_G(latent_size=feature_space, num_filters=filters)
netD = DCGAN_D(num_filters=filters)


def create_noise(b):
    """
    Create an image of random noise.
    """
    return Variable(torch.zeros(b, feature_space, 1, 1).normal_(0, 1))


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
            y_real = torch.ones(batch_size)
            y_fake = torch.zeros(batch_size)

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

            if x % 5 == 0:
                print('D loss: {}\tG loss: {}\tProgress: {}/{}'.format(D_train_loss, G_train_loss, x, 1875))

            x += 1


# train(1)
#
# torch.save(netG.state_dict(), "DCGAN1.pt")

netG.load_state_dict(torch.load('DCGAN1.pt'))

fig = plt.figure(figsize=(8, 8))
for i in range(20):
    img = netG(create_noise(20))
    arr = img.detach().numpy()
    fig.add_subplot(5, 4, i + 1)
    plt.imshow(arr[i][0], cmap='gray')

plt.show()
























