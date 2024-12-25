import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from astropy.io import fits
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class KeplerDataset(Dataset):
    def __init__(self, root, lightcurves: list, stage='fit', events='events.csv'):
        super().__init__()
        self.root = root
        self.lightcurves = lightcurves
        self.stage = stage
        if stage == 'fit':
            self.events = pd.read_csv(events)

    def __getitem__(self, index):
        try:
            with fits.open(os.path.join(self.root, self.lightcurves[index])) as f:
                lc_time, lc_flux = f[1].data['TIME'].astype(np.float32), f[1].data['PDCSAP_FLUX'].astype(np.float32)
                lc_flux = lc_flux / np.nanmedian(lc_flux)
                lc_mask = np.isnan(lc_flux)
                lc_time[lc_mask] = -1.0
                lc_mask = lc_mask.astype(np.float32)
        except Exception as e:
            print(e)
            print('{} load failed, please check'.format(os.path.join(self.root, self.lightcurves[index])))
            lc_time, lc_flux = np.zeros([400], dtype=np.float32), np.zeros([400], dtype=np.float32)
            lc_mask = np.ones([400], dtype=np.float32)
            # lc_index = torch.tensor(index, dtype=torch.int64)
            # return lc_time, lc_flux, lc_mask, lc_index
        
        if self.stage == 'fit':
            # label and flux are point-to-point
            # label[i] = 1 for flux[i] is part of a flare event
            # label[i] = 0 for flux[i] is not part of a flare event
            label = np.zeros(shape=lc_time.shape, dtype=np.int64)
            time_round = np.round(lc_time, 4)
            file = self.lightcurves[index].split('/')[-1]
            events_in_file = self.events[self.events['file'] == file]
            for i in range(events_in_file.shape[0]):
                start, end = events_in_file.iloc[i]['start'], events_in_file.iloc[i]['end']
                label[(time_round >= start) & (time_round <= end)] = 1.0
            return torch.tensor(lc_time), torch.tensor(lc_flux), torch.tensor(lc_mask), torch.tensor(label)
            
        elif self.stage == 'predict':
            # We need file name when we predict flares on a light curve.
            # Here we return the indedx so that we can get the file name when we predict.
            lc_index = torch.tensor(index, dtype=torch.int64)
            return torch.tensor(lc_time), torch.tensor(lc_flux), torch.tensor(lc_mask), lc_index

    def __len__(self):
        return len(self.lightcurves)
    
    def collate_fn(self, data):
        if self.stage == 'fit':
            lc_time, lc_flux, lc_mask, label = zip(*data)
            batch_time = pad_sequence(lc_time, batch_first=True)
            batch_flux = pad_sequence(lc_flux, batch_first=True)
            batch_mask = pad_sequence(lc_mask, batch_first=True, padding_value=1.0)
            batch_label = pad_sequence(label, batch_first=True)

            batch_flux = torch.where(torch.isnan(batch_flux), torch.full_like(batch_flux, 0.0), batch_flux)
            batch_data = torch.concat([batch_flux.unsqueeze(-1), batch_mask.unsqueeze(-1)], dim=-1)
            return batch_time, batch_data, batch_label
        
        elif self.stage == 'predict':
            lc_time, lc_flux, lc_mask, lc_index = zip(*data)
            batch_time = pad_sequence(lc_time, batch_first=True)
            batch_flux = pad_sequence(lc_flux, batch_first=True)
            batch_mask = pad_sequence(lc_mask, batch_first=True, padding_value=1.0)
            batch_index = torch.tensor(lc_index)

            batch_flux = torch.where(torch.isnan(batch_flux), torch.full_like(batch_flux, 0.0), batch_flux)
            batch_data = torch.concat([batch_flux.unsqueeze(-1), batch_mask.unsqueeze(-1)], dim=-1)
            return batch_time, batch_data, batch_index


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        lightcurves: str,
        events: str,
        val_size: float,
        batch_size: int,
        num_workers: int,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        file: str = self.hparams.lightcurves
        with open(file, 'r') as f:
            lightcurves = f.readlines()
        lightcurves = [lightcurve.rstrip('\n') for lightcurve in lightcurves]

        if stage == 'fit':
            train_lc, val_lc = train_test_split(lightcurves, test_size=self.hparams.val_size)

            self.train_set = KeplerDataset(self.hparams.data_root, train_lc, stage, self.hparams.events)
            self.val_set = KeplerDataset(self.hparams.data_root, val_lc, stage, self.hparams.events)

        elif stage == 'predict':
            self.predict_set = KeplerDataset(self.hparams.data_root, lightcurves, stage=stage)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
            shuffle=True, collate_fn=self.train_set.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
            shuffle=False, collate_fn=self.val_set.collate_fn
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
            shuffle=False, collate_fn=self.predict_set.collate_fn
        )
    

if __name__ == '__main__':
    datamodule = DataModule(
        data_root='F:/jmh/data/Yang_lightcurves/',
        lightcurves='./dataset/lightcurves.txt',
        events='./dataset/events.csv',
        val_size=0.2,
        batch_size=8,
        num_workers=0
    )
    datamodule.setup(stage='fit')
    for i, batch in enumerate(datamodule.train_dataloader()):
        print(batch)
        pass
