"""
BiLSTM Variational Autoencoder
-------------------------------

For production use (inference)

Trained using:
    - ABC Loss
    - Backbone: EfficientNet-B2
    - input_size: 1408
    - hidden_size: 512
    - latent_size: 256
    - num_layers: 1

Same architecture as training with minor adjustments for production.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class BiLSTMVariationalAutoencoder(nn.Module):
    """
    BiLSTM Variational Autoencoder
    -------------------------------

    Model architecture:

        Input (B, T, 1408)
          |
          v
        BiLSTM Encoder (bidirectional)
          |
          v
        fc_mu + fc_logvar
          |
          v
        Latent space (256)
          |
          v
        Expand (latent -> 1024)
          |
          v
        BiLSTM Decoder (bidirectional)
          |
          v
        Linear (1024 -> 1408)
          |
          v
        Reconstruction (B, T, 1408)

    Reparameterization trick is used only during training.
    During inference, we use only the mean (mu).
    """

    def __init__(
        self,
        input_size: int = 1408,
        hidden_size: int = 512,
        latent_size: int = 256,
        num_layers: int = 1
    ):
        """
        Initialize the model.

        Args:
            input_size: Feature size per frame (EfficientNet-B2 output)
            hidden_size: Hidden layer size in LSTM
            latent_size: Latent space dimension
            num_layers: Number of LSTM layers
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # =====================================================
        # Encoder - BiLSTM
        # =====================================================
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # BiLSTM output = hidden_size * 2 (forward + backward)
        encoder_output_size = hidden_size * 2

        # =====================================================
        # μ and log(σ²)
        # =====================================================
        self.fc_mu = nn.Linear(
            encoder_output_size,
            latent_size
        )

        self.fc_logvar = nn.Linear(
            encoder_output_size,
            latent_size
        )

        # =====================================================
        # Decoder
        # =====================================================
        self.expand = nn.Linear(
            latent_size,
            hidden_size * 2
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.output_layer = nn.Linear(
            hidden_size * 2,
            input_size
        )

        # =====================================================
        # State tracking
        # =====================================================
        self._is_training_mode = True

    # =========================================================
    # Reparameterization Trick
    # =========================================================
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick using standard normal distribution.

        Args:
            mu: Distribution mean [B, latent_size]
            logvar: Log variance [B, latent_size]

        Returns:
            z: Sampled latent vector [B, latent_size]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    # =========================================================
    # Encoder
    # =========================================================
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence into latent distribution.

        Args:
            x: [B, T, input_size] - Input feature sequence

        Returns:
            mu: [B, latent_size] - Distribution mean
            logvar: [B, latent_size] - Log variance
        """
        _, (hidden, _) = self.encoder(x)

        # hidden: (num_layers * 2, B, hidden_size)
        # For num_layers=1: (2, B, 512)
        forward_hidden = hidden[-2]   # Forward layer hidden state
        backward_hidden = hidden[-1]  # Backward layer hidden state

        # Combine both directions
        hidden_combined = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )  # (B, 1024)

        mu = self.fc_mu(hidden_combined)          # (B, 256)
        logvar = self.fc_logvar(hidden_combined)  # (B, 256)

        return mu, logvar

    # =========================================================
    # Decoder
    # =========================================================
    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation back to sequence.

        Args:
            latent: [B, latent_size] - Latent representation
            seq_len: Sequence length

        Returns:
            reconstruction: [B, seq_len, input_size] - Reconstructed sequence
        """
        # Expand latent space
        decoder_input = self.expand(latent)  # (B, 1024)

        # Repeat across sequence length
        decoder_input = decoder_input.unsqueeze(1)  # (B, 1, 1024)
        decoder_input = decoder_input.repeat(
            1, seq_len, 1
        )  # (B, seq_len, 1024)

        # Decode using BiLSTM
        decoder_output, _ = self.decoder(decoder_input)
        # (B, seq_len, 1024)

        # Project back to original feature size
        reconstruction = self.output_layer(decoder_output)
        # (B, seq_len, input_size)

        return reconstruction

    # =========================================================
    # Forward Pass
    # =========================================================
    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
        training_mode: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: [B, T, input_size] - Input feature sequence
            return_latent: Return latent variables if True
            training_mode: Training mode (if None, uses self.training)

        Returns:
            Dictionary containing:
                - reconstruction: [B, T, input_size] - Reconstructed sequence
                - mu: [B, latent_size] - Distribution mean (if return_latent=True)
                - logvar: [B, latent_size] - Log variance (if return_latent=True)
                - latent: [B, latent_size] - Sampled latent (if return_latent=True)
        """
        seq_len = x.size(1)

        # =====================================================
        # Encode
        # =====================================================
        mu, logvar = self.encode(x)

        # =====================================================
        # Reparameterization (training only)
        # =====================================================
        if training_mode is None:
            training_mode = self.training

        if training_mode:
            # Training: Use reparameterization trick
            latent = self.reparameterize(mu, logvar)
        else:
            # Inference: Use mean only (more stable)
            latent = mu

        # =====================================================
        # Decode
        # =====================================================
        reconstruction = self.decode(latent, seq_len)

        # =====================================================
        # Build output
        # =====================================================
        output = {
            "reconstruction": reconstruction
        }

        if return_latent:
            output["mu"] = mu
            output["logvar"] = logvar
            output["latent"] = latent

        return output

    # =========================================================
    # Inference Helper Methods
    # =========================================================
    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct sequence (fast inference).

        Args:
            x: [B, T, input_size] - Input feature sequence

        Returns:
            reconstruction: [B, T, input_size] - Reconstructed sequence
        """
        self.eval()
        output = self.forward(x, return_latent=False, training_mode=False)
        return output["reconstruction"]

    @torch.no_grad()
    def get_latent(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract latent representation only.

        Args:
            x: [B, T, input_size] - Input feature sequence

        Returns:
            latent: [B, latent_size] - Latent representation
        """
        self.eval()
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def get_reconstruction_error(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction error (MSE) per sample.

        Args:
            x: [B, T, input_size] - Input feature sequence

        Returns:
            error: [B] - Mean squared error per sample
        """
        self.eval()
        reconstruction = self.reconstruct(x)
        error = torch.mean(
            (reconstruction - x) ** 2,
            dim=(1, 2)
        )
        return error

    @torch.no_grad()
    def get_anomaly_score(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute anomaly score = 1 - exp(-error).

        Args:
            x: [B, T, input_size] - Input feature sequence

        Returns:
            anomaly_score: [B] - Between 0 and 1
                - Values near 0: Normal (real)
                - Values near 1: Anomalous (spoof)
        """
        self.eval()
        error = self.get_reconstruction_error(x)
        normal_prob = torch.exp(-error)      # Probability of being normal
        anomaly_score = 1.0 - normal_prob    # Probability of being anomalous
        return anomaly_score

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Classify samples as Real (0) or Spoof (1).

        Args:
            x: [B, T, input_size] - Input feature sequence
            threshold: Anomaly threshold value

        Returns:
            predictions: [B] - 0 for real, 1 for spoof
        """
        self.eval()
        anomaly_scores = self.get_anomaly_score(x)
        predictions = (anomaly_scores > threshold).long()
        return predictions


# =============================================================
# Model Loading Helpers
# =============================================================
def load_bilstm_vae(
    model_path: str,
    device: torch.device = None,
    input_size: int = 1408,
    hidden_size: int = 512,
    latent_size: int = 256,
    num_layers: int = 1
) -> BiLSTMVariationalAutoencoder:
    """
    Load BiLSTM-VAE model from weights file.

    Args:
        model_path: Path to weights file (.pth or .pt)
        device: Device (cpu/cuda)
        input_size: Input size
        hidden_size: Hidden layer size
        latent_size: Latent space size
        num_layers: Number of LSTM layers

    Returns:
        model: Loaded model in eval mode
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Create model
    model = BiLSTMVariationalAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers
    )

    # Load weights
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    # Handle full checkpoint files
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
