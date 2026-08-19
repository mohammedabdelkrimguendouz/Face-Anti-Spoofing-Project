import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM Undercomplete Autoencoder
    --------------------------------

    Architecture used during training:

        Input
          |
          v
        LSTM Encoder
          |
          v
        Linear
          |
          v
        Latent representation
          |
          v
        Linear
          |
          v
        LSTM Decoder
          |
          v
        Linear
          |
          v
        Reconstruction

    Training configuration:

        input_size = 1408
        hidden_size = 512
        latent_size = 256
        num_layers = 1

    The model was trained using ABC loss.
    """

    def __init__(
        self,
        input_size=1408,
        hidden_size=512,
        latent_size=256,
        num_layers=1,
        reverse_target=True,
        autoregressive_decode=True,
        teacher_forcing_ratio=0.5
    ):

        super().__init__()

        # =====================================================
        # Validate Undercomplete Autoencoder
        # =====================================================

        if latent_size >= input_size:

            raise ValueError(
                "This must be an UNDERcomplete autoencoder: "
                "latent_size must be < input_size."
            )

        # =====================================================
        # Configuration
        # =====================================================

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.latent_size = latent_size

        self.num_layers = num_layers

        self.reverse_target = reverse_target

        self.autoregressive_decode = (
            autoregressive_decode
        )

        self.teacher_forcing_ratio = (
            teacher_forcing_ratio
        )

        # =====================================================
        # Encoder
        # =====================================================

        self.encoder = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True

        )

        # =====================================================
        # Latent Projection
        #
        # hidden_size = 512
        #
        #        |
        #        v
        #
        # latent_size = 256
        # =====================================================

        self.latent = nn.Linear(

            hidden_size,

            latent_size

        )

        # =====================================================
        # Decoder Expansion
        #
        # latent_size = 256
        #
        #        |
        #        v
        #
        # hidden_size = 512
        # =====================================================

        self.expand = nn.Linear(

            latent_size,

            hidden_size

        )

        # =====================================================
        # Decoder
        # =====================================================

        if autoregressive_decode:

            self.decoder = nn.LSTM(

                input_size=(
                    input_size
                    + latent_size
                ),

                hidden_size=hidden_size,

                num_layers=num_layers,

                batch_first=True

            )

        else:

            self.decoder = nn.LSTM(

                input_size=hidden_size,

                hidden_size=hidden_size,

                num_layers=num_layers,

                batch_first=True

            )

        # =====================================================
        # Output Projection
        #
        # hidden_size = 512
        #
        #        |
        #        v
        #
        # input_size = 1408
        # =====================================================

        self.output_layer = nn.Linear(

            hidden_size,

            input_size

        )

    # =========================================================
    # Encoder
    # =========================================================

    def encode(self, x):

        """
        Encode input sequence into latent representation.

        Input:
            x: [B, T, 1408]

        Output:
            latent: [B, 256]
        """

        _, (hidden, cell) = self.encoder(x)

        # Last LSTM layer hidden state
        #
        # [num_layers, B, hidden_size]
        #
        # ->
        #
        # [B, hidden_size]

        hidden_last = hidden[-1]

        latent = self.latent(
            hidden_last
        )

        return latent

    # =========================================================
    # Repeat Vector Decoder
    # =========================================================

    def _decode_repeat_vector(
        self,
        latent,
        seq_len
    ):

        """
        Decode latent representation using
        repeated latent information.

        Input:
            latent: [B, 256]

        Output:
            reconstruction: [B, T, 1408]
        """

        decoder_input = self.expand(
            latent
        )

        # [B, 512]
        #
        # ->
        #
        # [B, 1, 512]

        decoder_input = (
            decoder_input.unsqueeze(1)
        )

        # Repeat for every timestep

        decoder_input = decoder_input.repeat(

            1,

            seq_len,

            1

        )

        # LSTM decoder

        decoder_output, _ = self.decoder(
            decoder_input
        )

        # [B, T, 512]
        #
        # ->
        #
        # [B, T, 1408]

        reconstruction = self.output_layer(
            decoder_output
        )

        return reconstruction

    # =========================================================
    # Autoregressive Decoder
    # =========================================================

    def _decode_autoregressive(
        self,
        latent,
        seq_len,
        target=None
    ):

        """
        Autoregressive decoder used during training.

        At every timestep:

            previous_frame + latent
                    |
                    v
                  LSTM
                    |
                    v
               reconstruction
                    |
                    v
              next timestep
        """

        batch_size = latent.size(0)

        device = latent.device

        # =====================================================
        # Initial hidden state
        # =====================================================

        h0 = self.expand(
            latent
        ).unsqueeze(0)

        # [1, B, 512]
        #
        # Repeat if num_layers > 1

        h0 = h0.repeat(

            self.num_layers,

            1,

            1

        )

        # Initial cell state

        c0 = torch.zeros_like(
            h0
        )

        hidden_state = (
            h0,
            c0
        )

        # =====================================================
        # Initial previous frame
        # =====================================================

        prev_frame = torch.zeros(

            batch_size,

            1,

            self.input_size,

            device=device

        )

        outputs = []

        # =====================================================
        # Autoregressive decoding
        # =====================================================

        for t in range(seq_len):

            # -------------------------------------------------
            # Concatenate:
            #
            # previous frame
            # +
            # latent
            #
            # [B, 1, 1408]
            # +
            # [B, 1, 256]
            #
            # =
            #
            # [B, 1, 1664]
            # -------------------------------------------------

            step_input = torch.cat(

                [
                    prev_frame,
                    latent.unsqueeze(1)
                ],

                dim=-1

            )

            # -------------------------------------------------
            # Decoder LSTM
            # -------------------------------------------------

            decoder_output, hidden_state = (
                self.decoder(
                    step_input,
                    hidden_state
                )
            )

            # -------------------------------------------------
            # Reconstruct current frame
            # -------------------------------------------------

            frame_output = self.output_layer(

                decoder_output

            )

            outputs.append(
                frame_output
            )

            # -------------------------------------------------
            # Teacher forcing
            #
            # Used ONLY during training.
            # -------------------------------------------------

            use_teacher_forcing = (

                self.training

                and target is not None

                and torch.rand(
                    1,
                    device=device
                ).item()
                < self.teacher_forcing_ratio

            )

            if use_teacher_forcing:

                prev_frame = target[
                    :,
                    t:t + 1,
                    :
                ]

            else:

                # Important:
                # During autoregressive decoding,
                # feed model prediction into next step.

                prev_frame = (
                    frame_output.detach()
                )

        # =====================================================
        # Combine timestep outputs
        # =====================================================

        reconstruction = torch.cat(

            outputs,

            dim=1

        )

        return reconstruction

    # =========================================================
    # Forward
    # =========================================================

    def forward(self, x):

        """
        Input:
            x = [B, 150, 1408]

        Output:
            reconstruction = [B, 150, 1408]
        """

        seq_len = x.size(1)

        # =====================================================
        # Encode
        # =====================================================

        latent = self.encode(x)

        # =====================================================
        # Target order
        #
        # Training used:
        #
        # reverse_target=True
        # =====================================================

        if self.reverse_target:

            target = torch.flip(

                x,

                dims=[1]

            )

        else:

            target = x

        # =====================================================
        # Decode
        # =====================================================

        if self.autoregressive_decode:

            reconstruction = (
                self._decode_autoregressive(

                    latent,

                    seq_len,

                    target

                )
            )

        else:

            reconstruction = (
                self._decode_repeat_vector(

                    latent,

                    seq_len

                )
            )

        # =====================================================
        # Restore original temporal order
        # =====================================================

        if self.reverse_target:

            reconstruction = torch.flip(

                reconstruction,

                dims=[1]

            )

        return reconstruction

# =============================================================
# ABC Loss
# =============================================================

def abc_loss(
    reconstruction,
    target,
    labels
):
    """
    ABC loss used during training.

    Input:

        reconstruction:
            [B, T, 1408]

        target:
            [B, T, 1408]

        labels:
            Real  = 0
            Spoof = 1

    ABC convention:

        Normal  = 1
        Anomaly = 0

    Therefore:

        y = 1 - labels

    Reconstruction error:

        L(x) = mean((reconstruction - target)^2)

    Normal probability:

        eta(x) = exp(-L(x))

    """

    # =========================================================
    # Reconstruction Error
    # =========================================================

    reconstruction_error = torch.mean(

        (reconstruction - target) ** 2,

        dim=(1, 2)

    )

    # =========================================================
    # Convert labels
    #
    # Your labels:
    #
    #   Real  = 0
    #   Spoof = 1
    #
    # ABC:
    #
    #   Normal  = 1
    #   Anomaly = 0
    # =========================================================

    y = 1.0 - labels.float()

    # =========================================================
    # Normal probability
    #
    # eta(x) = exp(-L(x))
    # =========================================================

    eta = torch.exp(
        -reconstruction_error
    )

    # =========================================================
    # Numerical stability
    # =========================================================

    eps = 1e-8

    eta = torch.clamp(

        eta,

        min=eps,

        max=1.0 - eps

    )

    # =========================================================
    # ABC Negative Log Likelihood
    # =========================================================

    loss = (

        y * reconstruction_error

        -

        (1.0 - y)
        * torch.log(1.0 - eta)

    )

    return loss.mean()