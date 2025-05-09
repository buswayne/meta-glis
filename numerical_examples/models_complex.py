import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, lat_lb=0.0, lat_ub=1.0):
        """
        Initialize the Encoder model.

        :param input_dim: Dimensionality of the input data.
        :param latent_dim: Dimensionality of the latent space (bottleneck).
        """
        super(Encoder, self).__init__()

        self.lat_lb = lat_lb
        self.lat_ub = lat_ub
        self.sigmoid = nn.Sigmoid()

        # Encoder: input_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Increased first hidden layer size
            nn.ReLU(),
            # nn.BatchNorm1d(256),  # Batch normalization for stable training
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout to reduce overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 32),  # Additional compression before latent space
            nn.ReLU(),
            nn.Linear(32, latent_dim)  # Bottleneck (latent space)
        )

    def forward(self, x):
        """
        Forward pass through the encoder.

        :param x: Input tensor.
        :return: Latent vector z (output of the encoder).
        """
        z = self.encoder(x)  # Encoding the input to latent space
        # Scale latent variable to [lb, ub]
        z_scaled = self.lat_lb + (self.lat_ub - self.lat_lb) * self.sigmoid(z)
        return z_scaled

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, bounds=None):
        """
        Initialize the Decoder model.

        :param latent_dim: Dimensionality of the latent space (input to the decoder).
        :param output_dim: Dimensionality of the output data (reconstructed input).
        """
        super(Decoder, self).__init__()

        if bounds is not None:
            # Extract bounds and register them
            lb, ub = zip(*bounds)
            self.register_buffer('out_lb', torch.tensor(lb, dtype=torch.float32))
            self.register_buffer('out_ub', torch.tensor(ub, dtype=torch.float32))
        else:
            self.out_lb = None
            self.out_ub = None

        self.sigmoid = nn.Sigmoid()

        # Decoder: latent_dim -> output_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),  # More gradual expansion from latent space
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),  # Batch normalization for stability
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout to improve generalization
            nn.Linear(128, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, output_dim)  # Final output layer
        )

    def forward(self, z):
        """
        Forward pass through the decoder.

        :param z: Latent vector (output of the encoder).
        :return: Reconstructed input (output of the decoder).
        """
        reconstructed_x = self.decoder(z)  # Decoding the latent space back to original space

        # Apply scaling if bounds are provided
        if self.out_lb is not None and self.out_ub is not None:
            reconstructed_x = self.out_lb + (self.out_ub - self.out_lb) * self.sigmoid(reconstructed_x)

        return reconstructed_x

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, lat_lb=0.0, lat_ub=1.0, out_bounds=None):
        """
        Initialize the Autoencoder model by combining the encoder and decoder.

        :param input_dim: Dimensionality of the input data.
        :param latent_dim: Dimensionality of the latent space.
        """
        super(Autoencoder, self).__init__()

        # Initialize Encoder and Decoder
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, lat_lb=lat_lb, lat_ub=lat_ub)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim, bounds=out_bounds)

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
