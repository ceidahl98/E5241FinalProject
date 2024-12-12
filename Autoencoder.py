import torch
import torch.nn as nn

def Activation():
    return nn.SiLU()  # Swish activation

def Norm(in_size):
    return nn.GroupNorm(num_groups=32, num_channels=in_size, eps=1e-6, affine=True)

class resNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.resBlock = nn.Sequential(
            Norm(in_channels),
            Activation(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            Norm(out_channels),
            Activation(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, groups=out_channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self.resBlock(x)

def downsample(in_channels, out_channels):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=4, stride=2, padding=1
    )

def upsample(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    )

class GaussianVAE(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.channels = [32, 64, 128, 128, 256,512, embedding_dim]

        # Input and output layers
        self.in_conv = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=3, stride=1, padding=1
        )
        self.encoder_layers.append(self.in_conv)

        self.out_conv = nn.Conv2d(
            self.channels[0], in_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Encoder layers
        num_layers = len(self.channels) - 1
        for i in range(num_layers):
            self.encoder_layers.append(
                nn.Sequential(
                    resNetBlock(self.channels[i], self.channels[i]),
                    resNetBlock(self.channels[i], self.channels[i]),
                    downsample(self.channels[i], self.channels[i + 1]),
                )
            )

        # Latent space: Mean and variance
        self.mean_layer = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        self.logvar_layer = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)

        # Decoder layers
        for i in range(num_layers - 1, -1, -1):
            self.decoder_layers.append(
                nn.Sequential(
                    upsample(self.channels[i + 1], self.channels[i]),
                    resNetBlock(self.channels[i], self.channels[i]),
                    resNetBlock(self.channels[i], self.channels[i]),
                )
            )

        self.decoder_layers.append(nn.Sequential(self.out_conv))

        # Combine encoder and decoder
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)  # Compute standard deviation
        epsilon = torch.randn_like(std)  # Sample epsilon from N(0, I)
        return mu + std * epsilon

    def forward(self, x):
        # Encode

        z_encoded = self.encoder(x)

        mu = self.mean_layer(z_encoded)
        logvar = self.logvar_layer(z_encoded)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decoder(z_encoded)
        return z,reconstruction, mu, logvar

    def encode(self, x):
        z_encoded = self.encoder(x)
        mu = self.mean_layer(z_encoded)
        logvar = self.logvar_layer(z_encoded)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)


