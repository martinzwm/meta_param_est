import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(dropout_rate)  # Adjusted dropout rate for example
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class Encoder(nn.Module):
    def __init__(self, dropout_rate):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        # self.resblock1 = ResidualBlock(16, dropout_rate)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        # self.resblock2 = ResidualBlock(32, dropout_rate)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2)
        # self.resblock3 = ResidualBlock(32, dropout_rate)

        self.linear1 = nn.Linear(32 * 8 * 8, 512)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        # x = self.resblock1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # x = self.resblock2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # x = self.resblock3(x)

        x = x.view(-1, 32 * 8 * 8)
        x = self.linear1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dropout_rate):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(512, 32 * 8 * 8)
        self.bn0 = nn.BatchNorm2d(32)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # self.resblock1 = ResidualBlock(32, dropout_rate)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        # self.resblock2 = ResidualBlock(16, dropout_rate)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(2)
        # self.resblock3 = ResidualBlock(2, dropout_rate)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = x.view(-1, 32, 8, 8)
        x = self.bn0(x)

        x = self.upsample1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.resblock1(x)

        x = self.upsample2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.resblock2(x)

        x = self.upsample3(x)
        x = F.relu(self.conv3(x))
        # x = self.resblock3(x)
        return x
    

class Autoencoder(nn.Module):
    def __init__(self, dropout_rate):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(dropout_rate)
        self.decoder = Decoder(dropout_rate)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LinearAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(2 * 64 * 64, latent_dim)
        self.decoder = nn.Linear(latent_dim, 2 * 64 * 64)
        # self.encoder = nn.Sequential(
        #     nn.Linear(2 * 64 * 64, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(2048, 512),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(512, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(2048, 2 * 64 * 64),
        # )
        
    def forward(self, x):
        x = x.view(-1, 2 * 64 * 64)
        x = self.encoder(x)

        x = self.decoder(x)
        x = x.view(-1, 2, 64, 64)
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, latent_dim=512, normalize: bool = False):
        r'''
        latent_dim (int): Dimension of latent space
        normalize (bool): Whether to restrict the output latent onto the unit hypersphere
        '''
        super(SimpleEncoder, self).__init__()

        self.conv1 = nn.Conv2d(2, 32, 4, stride=2, padding=1) # 64x64 --> 32x32
        self.conv2 = nn.Conv2d(32, 32*2, 4, stride=2, padding=1) # 32x32 --> 16x16
        self.conv3 = nn.Conv2d(32*2, 32*4, 4, stride=2, padding=1) # 16x16 --> 8x8
        self.conv4 = nn.Conv2d(32*4, 32*8, 4, stride=2, padding=1) # 8x8 --> 4x4
        self.conv5 = nn.Conv2d(32*8, 32*16, 4, stride=2, padding=1) # 4x4 --> 2x2
        self.conv6 = nn.Conv2d(32*16, latent_dim, 4, stride=2, padding=1) # 2x2 --> 1x1
        self.fc = nn.Linear(latent_dim, latent_dim)

        self.nonlinearity = nn.ReLU()
        self.normalize = normalize

    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x))
        x = self.nonlinearity(self.conv6(x).flatten(1))
        x = self.fc(x)
        if self.normalize:
            x = F.normalize(x)
        return x
    
    
class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim):
        r'''
        latent_dim (int): Dimension of latent space
        '''
        super(SimpleDecoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(latent_dim, 32*16, 4, stride=2, padding=1) # 1x1 --> 2x2
        self.conv2 = nn.ConvTranspose2d(32*16, 32*8, 4, stride=2, padding=1) # 2x2 --> 4x4
        self.conv3 = nn.ConvTranspose2d(32*8, 32*4, 4, stride=2, padding=1) # 4x4 --> 8x8
        self.conv4 = nn.ConvTranspose2d(32*4, 32*2, 4, stride=2, padding=1) # 8x8 --> 16x16
        self.conv5 = nn.ConvTranspose2d(32*2, 32, 4, stride=2, padding=1) # 16x16 --> 32x32
        self.conv6 = nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1) # 32x32 --> 64x64
        self.nonlinearity = nn.ReLU()

    def forward(self, z):
        z = z[..., None, None]  # make it convolution-friendly
        x = self.nonlinearity(self.conv1(z))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x))
        return self.conv6(x)
    

class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim, normalize: bool = False):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = SimpleEncoder(latent_dim)
        self.decoder = SimpleDecoder(latent_dim)

    def forward(self, x):
        emb = self.encoder(x)
        recon_x = self.decoder(emb)
        return recon_x


class DummyAutoencoder(nn.Module):
    def __init__(self):
        super(DummyAutoencoder, self).__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def forward(self, x):
        return x


class TestModel:

    def __init__(self):
        pass

    def test_autoencoder(self):
        autoencoder = Autoencoder(dropout_rate=0.1)
        input_image = torch.rand(10, 2, 64, 64)
        output_image = autoencoder(input_image)
        print(output_image.shape)
        print(sum(p.numel() for p in autoencoder.parameters()))

    def test_linear_autoencoder(self):
        autoencoder = LinearAutoencoder(latent_dim=512)
        input_image = torch.rand(10, 2, 64, 64)
        output_image = autoencoder(input_image)
        print(output_image.shape)
        print(sum(p.numel() for p in autoencoder.parameters()))


    def test_simple_autoencoder(self):
        autoencoder = SimpleAutoencoder(latent_dim=512, normalize=True)
        input_image = torch.rand(10, 2, 64, 64)
        output_image = autoencoder(input_image)
        print(output_image.shape)
        print(sum(p.numel() for p in autoencoder.parameters()))

    
    def test_dummy_autoencoder(self):
        autoencoder = DummyAutoencoder()
        input_image = torch.rand(10, 2, 64, 64)
        output_image = autoencoder(input_image)
        print(output_image.shape)
        print(sum(p.numel() for p in autoencoder.parameters()))


if __name__ == '__main__':
    unittest = TestModel()
    # unittest.test_autoencoder()
    # unittest.test_linear_autoencoder()
    # unittest.test_simple_autoencoder()
    # unittest.test_dummy_autoencoder()