from torch import nn
import torch
import datetime

"""
Model used to train multitask regression (or if you modify it)
classification problems for 1d signals.
"""


class SignalCNNTransformer(nn.Module):
    def __init__(
        self,
        input_channels,
        conv_channels,
        ff_layers_dim,
        output_size,
        downsample_factor=1,
        resolution=10000,
        batch_norm=False,
        n_heads=10,
        encoder_layers=1,
        kernel_downsample_size=5,
        uses_coupling_guess=False,
    ):
        super(SignalCNNTransformer, self).__init__()

        ## Hyperparameters ##
        #####################]
        self.name = "SignalCNNTransformer" + datetime.now().strftime("%Y%m%d_%H%M%S")

        self.layer_norm = batch_norm
        self.input_channels = input_channels
        self.uses_coupling_guess = uses_coupling_guess

        ## Downsample Block ##
        ######################
        dummy_input = torch.randn(1, 2, resolution)

        self.downsample_block = nn.ModuleList()
        self.downsample_channels = [2**n for n in range(1, downsample_factor + 1, 1)]
        print(f"Downsample Channels: {self.downsample_channels}")

        for c1, c2 in zip(self.downsample_channels, self.downsample_channels[1:]):
            self.downsample_block.append(
                nn.Conv1d(
                    c1, c2, kernel_size=kernel_downsample_size, stride=1, padding=0
                )
            )
            # c2, length
            if self.layer_norm:
                self.downsample_block.append(nn.BatchNorm1d(c2))
            self.downsample_block.append(nn.GELU())
            self.downsample_block.append(nn.MaxPool1d(5, 2))

        self.downsample_block = nn.Sequential(*self.downsample_block)

        self.downsampled_length = self.downsample_block(dummy_input).shape[-1]

        print(
            f"Resolution size of {resolution} gets downsampled to {self.downsampled_length}\n"
        )

        ## Convolutional Layer Block ##
        ###############################
        conv_channels = [self.downsample_channels[-1]] + conv_channels
        print(f"Conv Channels: {conv_channels[1:]}")

        self.conv_layers = nn.ModuleList()
        for c1, c2 in zip(conv_channels, conv_channels[1:]):
            self.conv_layers.append(
                nn.Conv1d(c1, c2, kernel_size=3, stride=1, padding=1),
            )
            if self.layer_norm:
                self.conv_layers.append(nn.BatchNorm1d(c2))
            self.conv_layers.append(nn.GELU())

        self.conv_layers = nn.Sequential(*self.conv_layers)
        downsampled_shape = self.conv_layers(self.downsample_block(dummy_input)).shape[
            -1
        ]
        print(
            f"Resolution size of {resolution} gets downsampled to {downsampled_shape}\n"
        )

        ## Transformer Layer ##
        #######################
        # self.cls = torch.nn.Parameter(torch.randn(1, 1, conv_channels[-1]))

        self.cls = torch.nn.Parameter(torch.randn(1, conv_channels[-1], 1))

        print(f"CLS Token size: {self.cls.shape}\n")

        # Layer Norm the CLS token so it doesn't blow up.
        self.layer_norm = torch.nn.LayerNorm(
            conv_channels[-1], elementwise_affine=False
        )

        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=conv_channels[-1],  # +1 from cls token.
            nhead=n_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.transformer_layer, num_layers=encoder_layers
        )

        ## Feed Forward Block ##
        ########################

        self.ff_layers_dim = [conv_channels[-1]] + ff_layers_dim
        print(f"FF Channels: {ff_layers_dim}")

        self.ff_layers = nn.ModuleList()

        for c1, c2 in zip(self.ff_layers_dim, self.ff_layers_dim[1:]):
            self.ff_layers.append(nn.Linear(c1, c2))
            self.ff_layers.append(nn.GELU())
        self.ff_layers = nn.Sequential(*self.ff_layers)

        ## Param FF Layer ##
        ####################
        if self.uses_coupling_guess:
            # We need to add extra neuron for the coupling guess.
            self.param_layers = [nn.Linear(self.ff_layers_dim[-1] + 1, output_size)]
        else:
            self.param_layers = [nn.Linear(self.ff_layers_dim[-1], output_size)]
        # self.param_layers.append(nn.ReLU()) # Force positive Values
        self.param_layers = nn.Sequential(*self.param_layers)

        self.hyperparameters = {
            "Batch Norm": batch_norm,
            "Downsample Channels": str(self.downsample_channels),
            "Input Channels": input_channels,
            "Conv Channel Layers": conv_channels,
            "Feed Forward Layers": str([downsampled_shape + 1] + ff_layers_dim),
            "Output Size": output_size,
            "Encoder Layers": encoder_layers,
            "Encoder Heads": n_heads,
            "CLS Token Shape": self.cls.shape,
            "Kernel Downsample Size": kernel_downsample_size,
        }

    def forward(self, x):
        # Downsample intial input
        x = self.downsample_block(x)

        # Conv Layers
        x = self.conv_layers(x)

        # Add CLS token
        x = torch.cat((x, self.cls.repeat(x.shape[0], 1, 1)), dim=-1)

        # Pass Conv channels to encoder
        x = torch.transpose(x, 1, 2)

        # Normalize CLS token to prevent blowup
        x = self.layer_norm(x)

        ## Transformer Encoder Layer ##
        x = self.transformer_encoder(x)

        # Rip out cls token
        x = x[:, -1, :]

        # Feed Forward Layers and output
        x = self.ff_layers(x)

        # Spit out parameters
        x = self.param_layers(x)

        return x
