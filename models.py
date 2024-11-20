import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        Initialize the Encoder model.

        :param input_dim: Dimensionality of the input data.
        :param latent_dim: Dimensionality of the latent space (bottleneck).
        """
        super(Encoder, self).__init__()

        # Encoder: input_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),    # First hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),           # Second hidden layer
            nn.ReLU(),
            nn.Linear(64, latent_dim)     # Bottleneck (latent space)
        )

    def forward(self, x):
        """
        Forward pass through the encoder.

        :param x: Input tensor.
        :return: Latent vector z (output of the encoder).
        """
        z = self.encoder(x)  # Encoding the input to latent space
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        """
        Initialize the Decoder model.

        :param latent_dim: Dimensionality of the latent space (input to the decoder).
        :param output_dim: Dimensionality of the output data (reconstructed input).
        """
        super(Decoder, self).__init__()

        # Decoder: latent_dim -> output_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),    # First hidden layer
            nn.ReLU(),
            nn.Linear(64, 128),           # Second hidden layer
            nn.ReLU(),
            nn.Linear(128, output_dim),   # Output layer (reconstruct the original input)
        )

    def forward(self, z):
        """
        Forward pass through the decoder.

        :param z: Latent vector (output of the encoder).
        :return: Reconstructed input (output of the decoder).
        """
        reconstructed_x = self.decoder(z)  # Decoding the latent space back to original space
        return reconstructed_x

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        Initialize the Autoencoder model by combining the encoder and decoder.

        :param input_dim: Dimensionality of the input data.
        :param latent_dim: Dimensionality of the latent space.
        """
        super(Autoencoder, self).__init__()

        # Initialize Encoder and Decoder
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim)

    def forward(self, x):
        """
        Forward pass through the Autoencoder: input -> Encoder -> Latent Space -> Decoder -> Reconstructed Input.

        :param x: Input tensor (original data).
        :return: Reconstructed input after passing through encoder and decoder.
        """
        # Pass through encoder to get the latent space representation
        z = self.encoder(x)

        # Pass the latent representation through the decoder to get the reconstruction
        reconstructed_x = self.decoder(z)

        return reconstructed_x
