from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_fcn4flare import FCN4FlareConfig


class MaskDiceLoss(nn.Module):
    r"""
    Computes the Mask Dice Loss between the predicted and target tensors.
    $$
    \text{loss} = 1 - \frac{2 \times \text{intersection} + \epsilon}{\text{predicted} + \text{target} + \epsilon}
    $$

    Args:
        maskdice_threshold (float): Threshold value for the predicted tensor.

    Returns:
        loss (float): Computed Mask Dice Loss.
    """
    def __init__(self, maskdice_threshold):
        super().__init__()
        self.maskdice_threshold = maskdice_threshold

    def forward(self, inputs, targets):
        """
        Computes the forward pass of the Mask Dice Loss.

        Args:
            inputs (torch.Tensor): Predicted tensor.
            targets (torch.Tensor): Target tensor.

        Returns:
            loss (float): Computed Mask Dice Loss.
        """
        n = targets.size(0)
        smooth = 1e-8
        
        # Apply thresholding to inputs
        inputs_act = torch.gt(inputs, self.maskdice_threshold)
        inputs_act = inputs_act.long()
        inputs = inputs * inputs_act
        
        intersection = inputs * targets
        dice_diff = (2 * intersection.sum(1) + smooth) / (inputs.sum(1) + targets.sum(1) + smooth * n)
        loss = 1 - dice_diff.mean()
        return loss


class NaNMask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        # Create a mask where NaNs are marked as 1
        nan_mask = torch.isnan(inputs).float()
        # Replace NaNs with 0 in the input tensor
        inputs = torch.nan_to_num(inputs, nan=0.0)
        # Concatenate the input tensor with the NaN mask
        return torch.cat([inputs, nan_mask], dim=-1)


class SamePadConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            input_dim, output_dim, kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.batchnorm = nn.BatchNorm1d(output_dim)
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.gelu(x)
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super().__init__()
        self.conv1 = SamePadConv(input_dim, output_dim, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(output_dim, output_dim, kernel_size, dilation=dilation)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class Backbone(nn.Module):
    def __init__(self, input_dim, dim_list, dilation, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                dim_list[i-1] if i > 0 else input_dim,
                dim_list[i],
                kernel_size=kernel_size,
                dilation=dilation[i]
            )
            for i in range(len(dim_list))
        ])
        
    def forward(self, x):
        return self.net(x)


class LightCurveEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, depth, dilation):
        super().__init__()
        self.mapping = nn.Conv1d(input_dim + 1, output_dim, 1)  # +1 for NaN mask
        self.backbone = Backbone(
            output_dim,
            [output_dim] * depth,
            dilation,
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        x = x.transpose(1, 2)   # B x Ci x T
        x = self.mapping(x)     # B x Ch x T
        x = self.backbone(x)    # B x Co x T
        x = self.repr_dropout(x)
        return x


class SegHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = SamePadConv(input_dim, input_dim, 3)
        self.projector = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        # x: B x Ci x T
        x = self.conv(x)       # B x Ci x T
        x = self.projector(x)  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        return x
    

class FCN4FlarePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """
    config_class = FCN4FlareConfig
    base_model_prefix = "fcn4flare"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


@dataclass
class FCN4FlareOutput(ModelOutput):
    """
    Output type of FCN4Flare.

    Args:
        loss (`Optional[torch.FloatTensor]` of shape `(1,)`, *optional*):
            Mask Dice loss if labels provided, None otherwise.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, output_dim)`):
            Prediction scores of the model.
        hidden_states (`torch.FloatTensor` of shape `(batch_size, hidden_dim, sequence_length)`):
            Hidden states from the encoder.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None


class FCN4FlareModel(FCN4FlarePreTrainedModel):
    def __init__(self, config: FCN4FlareConfig):
        super().__init__(config)
        
        self.nan_mask = NaNMask()
        self.encoder = LightCurveEncoder(
            config.input_dim,
            config.hidden_dim,
            config.depth,
            config.dilation
        )
        self.seghead = SegHead(config.hidden_dim, config.output_dim)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features,
        sequence_mask=None,
        labels=None,
        return_dict=True,
    ):
        # Apply NaN masking
        inputs_with_mask = self.nan_mask(input_features)

        # Encoder and segmentation head
        outputs = self.encoder(inputs_with_mask)
        logits = self.seghead(outputs)
        
        # Loss calculation
        loss = None
        if labels is not None:
            loss_fct = MaskDiceLoss(self.config.maskdice_threshold)
            logits_sigmoid = torch.sigmoid(logits).squeeze(-1)
            
            if sequence_mask is not None:
                # Copy labels and replace padding positions with zeros
                labels_for_loss = labels.clone()
                labels_for_loss = torch.nan_to_num(labels_for_loss, nan=0.0)
                labels_for_loss = labels_for_loss * sequence_mask
                logits_sigmoid = logits_sigmoid * sequence_mask
                loss = loss_fct(logits_sigmoid, labels_for_loss)
            else:
                loss = loss_fct(logits_sigmoid, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return FCN4FlareOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs
        )
