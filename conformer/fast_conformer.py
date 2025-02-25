import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .modules import Linear
from .convolution import FastConformerSubsampling
from .encoder import ConformerEncoder


class FastConformerConvModule(nn.Module):
    """
    Fast Conformer convolution module with reduced kernel size (9 instead of 31)
    as described in the Fast Conformer paper.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 9,  # Reduced from 31 to 9
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        from .convolution import ConformerConvModule
        super(FastConformerConvModule, self).__init__()
        
        # Use the original ConformerConvModule but with smaller kernel size
        self.conv_module = ConformerConvModule(
            in_channels=in_channels,
            kernel_size=kernel_size,
            expansion_factor=expansion_factor,
            dropout_p=dropout_p
        )
    
    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv_module(inputs)


class FastConformerEncoder(nn.Module):
    """
    Fast Conformer Encoder with 8x downsampling and modified convolutional modules.
    
    Args:
        input_dim (int): Dimension of input vector
        encoder_dim (int): Dimension of encoder output
        num_layers (int): Number of encoder layers
        num_attention_heads (int): Number of attention heads
        feed_forward_expansion_factor (int): Expansion factor of feed forward module
        conv_expansion_factor (int): Expansion factor of conformer convolution module
        input_dropout_p (float): Dropout probability of inputs
        feed_forward_dropout_p (float): Dropout probability of feed forward module
        attention_dropout_p (float): Dropout probability of attention module
        conv_dropout_p (float): Dropout probability of conformer convolution module
        conv_kernel_size (int): Kernel size of conformer convolution module
        half_step_residual (bool): Flag indication whether to use half step residual or not
    """
    def __init__(
            self,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 9,  # Reduced from 31 to 9
            half_step_residual: bool = True,
    ) -> None:
        super(FastConformerEncoder, self).__init__()
        
        # Use FastConformerSubsampling with 8x downsampling
        self.subsampling = FastConformerSubsampling(1, encoder_dim)
        
        # Use the original ConformerEncoder but pass custom conv_module_class
        from .encoder import ConformerBlock
        
        # Override the ConformerBlock to use FastConformerConvModule
        class FastConformerBlock(ConformerBlock):
            def __init__(self, *args, **kwargs):
                super(FastConformerBlock, self).__init__(*args, **kwargs)
                # Replace the conv_module with FastConformerConvModule
                self.conv_module = FastConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                )
        
        self.layers = nn.ModuleList([
            FastConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual,
            ) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(p=input_dropout_p)
    
    def count_parameters(self) -> int:
        """Count number of trainable parameters in encoder"""
        num_parameters = 0
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
        return num_parameters
    
    def update_dropout(self, dropout_p: float) -> None:
        """Update dropout probability of encoder"""
        for layer in self.layers:
            layer.feed_forward_module.dropout.p = dropout_p
            layer.attention_module.dropout.p = dropout_p
            layer.conv_module.conv_module.sequential[-1].p = dropout_p
    
    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for encoder training.
        
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        
        Returns:
            (Tensor, Tensor)
            
            * outputs: A output sequence of encoder. `FloatTensor` of size ``(batch, seq_length, dimension)``
            * output_lengths: The length of encoder outputs. ``(batch)``
        """
        outputs, output_lengths = self.subsampling(inputs, input_lengths)
        outputs = self.dropout(outputs)
        
        for layer in self.layers:
            outputs = layer(outputs)
            
        return outputs, output_lengths


class FastConformer(nn.Module):
    """
    Fast Conformer: Conformer with fast downsampling and reduced kernel size
    
    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of encoder output
        num_encoder_layers (int, optional): Number of encoder layers
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of convolution module
        input_dropout_p (float, optional): Dropout probability of inputs
        feed_forward_dropout_p (float, optional): Dropout probability of feed forward module
        attention_dropout_p (float, optional): Dropout probability of attention module
        conv_dropout_p (float, optional): Dropout probability of convolution module
        conv_kernel_size (int, optional): Kernel size of convolution module
        half_step_residual (bool, optional): Flag indication whether to use half step residual or not
        
    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
        
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 9,  # Reduced from 31 to 9 as per the paper
            half_step_residual: bool = True,
    ) -> None:
        super(FastConformer, self).__init__()
        self.encoder = FastConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.fc = Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        return outputs, encoder_output_lengths