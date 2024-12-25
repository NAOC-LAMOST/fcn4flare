from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import dice
import numpy as np
import pandas as pd
import os



class MaskDiceLoss(nn.Module):
    """
    Computes the Mask Dice Loss between the predicted and target tensors.

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
        inputs_act = torch.gt(inputs, self.maskdice_threshold)
        inputs_act = inputs_act.long()
        inputs = inputs * inputs_act
        intersection = inputs * targets
        dice_diff = (2 * intersection.sum(1) + smooth) / (inputs.sum(1) + targets.sum(1) + smooth * n)
        loss = 1 - dice_diff.mean()
        return loss



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
        self.mapping = nn.Conv1d(input_dim, output_dim, 1)
        self.backone = Backbone(
            output_dim,
            [output_dim] * depth,
            dilation,
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        # x: B x T x Ci
        x = x.transpose(1, 2)   # B x Ci x T
        x = self.mapping(x)     # B x Ch x T
        x = self.backone(x)     # B x Co x T
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


class FCN4Flare(pl.LightningModule):
    """
    A PyTorch Lightning module for the FCN4Flare model.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden features.
        output_dim (int): The number of output features.
        depth (int): The depth of the backbone network.
        dilation (list): A list of dilation rates for the backbone network.
        maskdice_threshold (float): The threshold for the MaskDiceLoss function.

    Attributes:
        encoder (LightCurveEncoder): The encoder network.
        seghead (SegHead): The segmentation head network.
        loss_fn (MaskDiceLoss): The loss function.

    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        dilation: list,
        maskdice_threshold: float
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = LightCurveEncoder(
            self.hparams.input_dim,
            self.hparams.hidden_dim,
            self.hparams.depth,
            self.hparams.dilation
        )
        self.seghead = SegHead(self.hparams.hidden_dim, self.hparams.output_dim)

        self.loss_fn = MaskDiceLoss(self.hparams.maskdice_threshold)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.seghead(x)
        return x

    def training_step(self, batch, batch_idx):
        batch_time, batch_data, batch_label = batch
        logits = torch.sigmoid(self(batch_data)).squeeze(-1)
        loss = self.loss_fn(logits, batch_label)
        pred = torch.gt(logits, 0.5).long()
        train_dice = dice(pred, batch_label, ignore_index=0)
        self.log('train_loss', loss)
        self.log('train_dice', train_dice, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_time, batch_data, batch_label = batch
        logits = torch.sigmoid(self(batch_data)).squeeze(-1)
        loss = self.loss_fn(logits, batch_label)
        pred = torch.gt(logits, 0.5).long()
        val_dice = dice(pred, batch_label, ignore_index=0)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_dice', val_dice, sync_dist=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        """
        Runs a single prediction step on a batch of data.

        Args:
            batch: A tuple containing the batch time, batch data, and batch index.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the light curve index, start time, and end time of each detected flare event.
        """
        batch_time, batch_data, batch_index = batch

        # forward pass
        logits = torch.sigmoid(self(batch_data)).squeeze(-1)
        pred = torch.gt(logits, 0.5).long()

        batch_time = batch_time.cpu().numpy()
        batch_data = batch_data.cpu().numpy()
        batch_index = batch_index.cpu().numpy()
        pred = pred.cpu().numpy()
        batch_flux = batch_data[:, :, :1]
        
        # extract flare events in batch
        lc_index_list = []  # list of light curves index in batch
        start_list =  []    # list of start time of flare in batch
        end_list = []       # list of end time of flare in batch

        for i in range(pred.shape[0]):
            flux = batch_flux[i]
            file_pred = pred[i][:]
            flare_idx = np.argwhere(file_pred == 1)

            # There are no flare on flux[i]
            if flare_idx.shape[0] == 0:
                continue

            # Two pointers algorithm to find start and end of flare
            # flare_idx of a flare should be continual.
            # For example: flare_idx = [0, 1, 2, 8, 9, 10]
            # there are two flares according to flare_idx,
            # [0, 1, 2] represents a flare and [8, 9, 10] represents the other.
            left, right = 0, 0
            while right < flare_idx.shape[0]:
                if left >= right:
                    right += 1
                    continue

                # Note: flare_idx which is get from np.argwhere() is a tuple.
                if flare_idx[right][0] - flare_idx[right - 1][0] > 1 or right == flare_idx.shape[0] - 1:
                    start_idx, end_idx = flare_idx[left][0], flare_idx[right - 1][0]
                    peak_idx = np.argmax(flux[start_idx : end_idx+1]) + start_idx

                    # If start time is the peak time, flux[start] must be greater than flux[start-1].
                    # If not, current event should be skipped.
                    if peak_idx == start_idx and flux[peak_idx] <= flux[peak_idx - 1]:
                        left = right
                        continue

                    # Duration of a flare should be longer than 3 cadances
                    if end_idx - start_idx < 2:
                        left = right
                        continue

                    # Time for flux to decrease should be longer than that to increase.
                    if end_idx - peak_idx <= peak_idx - start_idx:
                        left = right
                        continue

                    # alter
                    try:
                        alter = (flux[peak_idx] - flux[start_idx - 2]) / (flux[peak_idx] - flux[end_idx + 2] + 1e-8)
                    except:
                        left = right
                        continue
                    # flare 前后 flux 水平应该差不多
                    if alter < 0.5 or alter > 2 or np.isnan(alter):
                        left = right
                        continue
                    
                    # peak 前的斜率要大于无flare时的斜率
                    if np.abs(flux[peak_idx] - flux[peak_idx-1]) > 2 * np.abs(flux[peak_idx-1] - flux[peak_idx-2]):
                        left = right
                        continue
                    
                    lc_index = batch_index[i]
                    start, end = batch_time[i][start_idx], batch_time[i][end_idx]
                    lc_index_list.append(lc_index)
                    start_list.append(start)
                    end_list.append(end)

                    left = right
                
                right += 1

        return {'lc_index': lc_index_list, 'start': start_list, 'end': end_list}

    def on_predict_epoch_end(self, results):
        """
        A PyTorch Lightning hook that is called at the end of the predict epoch.

        Args:
            results (list): A list of dictionaries containing the predicted flare start and end indices for each light curve.

        Returns:
            None
        """
        lc_index_list, start_list, end_list = [], [], []
        for result in results[0]:
            lc_index_list.extend(result['lc_index'])
            start_list.extend(result['start'])
            end_list.extend(result['end'])
        
        df = pd.DataFrame({
            'lc_index': lc_index_list, 
            'start': start_list,
            'end': end_list
        })

        if not os.path.exists('results'):
            os.mkdir('results')
            
        df.to_csv('results/flare_predict.csv', index=False)
