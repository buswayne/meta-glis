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

        # Apply scaling if bounds are provided
        if self.out_lb is not None and self.out_ub is not None:
            reconstructed_x = self.out_lb + (self.out_ub - self.out_lb) * self.sigmoid(reconstructed_x)

        return reconstructed_x

class DecoderTrivial(nn.Module):
    def __init__(self, latent_dim, output_dim, bounds=None):
        """
        Initialize the Decoder model.

        :param latent_dim: Dimensionality of the latent space (input to the decoder).
        :param output_dim: Dimensionality of the output data (reconstructed input).
        """
        super(DecoderTrivial, self).__init__()

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
            nn.Linear(latent_dim, output_dim),   # Output layer (reconstruct the original input)
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

class AutoencoderTrivial(nn.Module):
    def __init__(self, input_dim, latent_dim, lat_lb=0.0, lat_ub=1.0, out_bounds=None):
        """
        Initialize the Autoencoder model by combining the encoder and decoder.

        :param input_dim: Dimensionality of the input data.
        :param latent_dim: Dimensionality of the latent space.
        """
        super(AutoencoderTrivial, self).__init__()

        # Initialize Encoder and Decoder
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, lat_lb=lat_lb, lat_ub=lat_ub)
        self.decoder = DecoderTrivial(latent_dim=latent_dim, output_dim=input_dim, bounds=out_bounds)

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
