# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import h5py

from dataclasses import asdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict

from .config_torchwavenet import Config
from .dataset import MultiChannelDataset
from .torchwavenet import Wave

_map = lambda x: torch.view_as_real(x).transpose(-2,-1)
_map2 = lambda x: torch.flatten(x, start_dim=1, end_dim=2)

class WaveLearner:
    def __init__(self, cfg: Config, model: nn.Module, gen):
        self.cfg = cfg
        self.gen = gen

        # Store some import variables
        self.model_dir = cfg.model_dir
        self.log_every = cfg.trainer.log_every
        self.validate_every = cfg.trainer.validate_every
        self.save_every = cfg.trainer.save_every
        self.max_steps = cfg.trainer.max_steps
        self.build_dataloaders()

        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.trainer.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min",
        )
        self.autocast = torch.cuda.amp.autocast(enabled=cfg.trainer.fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.trainer.fp16)
        self.step = 0

        self.loss_fn = nn.MSELoss()
        self.writer = SummaryWriter(self.model_dir)

    def build_dataloaders(self):
        cdt = self.cfg.data
        with h5py.File(self.cfg.data.data_dir,'r') as data_h5file:
            sig_data = np.array(data_h5file.get('dataset'))
        with h5py.File(self.cfg.data.val_data_dir,'r') as data_h5file:
            valid_data = np.array(data_h5file.get('dataset'))
        self.train_dataset = MultiChannelDataset(
            self.gen, sig_data, cdt.sig_len, cdt.num_ant, cdt.batch_size*100, cdt.sinr_range, cdt.soi_aoa, fix=False
        )
        self.val_dataset = MultiChannelDataset(
            self.gen, valid_data, cdt.sig_len, cdt.num_ant, cdt.batch_size*64, cdt.sinr_range, cdt.soi_aoa, fix=True#TODO
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cdt.batch_size,
            shuffle=False,
            num_workers=cdt.num_workers,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=cdt.batch_size * 4,
            shuffle=False,
            num_workers=cdt.num_workers,
            pin_memory=True,
        )

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) 
                      else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) 
                          else v for k, v in self.optimizer.state_dict().items()},
            'cfg': asdict(self.cfg),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self):

        while True:
            for features in tqdm(
                self.train_dataloader, 
                desc=f"Training ({self.step} / {self.max_steps})"
            ):
                loss = self.train_step(features)

                # Check for NaNs
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at step {self.step}.')

                if self.step % self.log_every == 0:
                    self.writer.add_scalar('train/loss', loss, self.step)
                    self.writer.add_scalar(
                        'train/grad_norm', self.grad_norm, self.step)
                if self.step % self.save_every == 0:
                    self.save_to_checkpoint()

                if self.step % self.validate_every == 0:
                    val_loss = self.validate()
                    # Update the learning rate if it plateus
                    self.lr_scheduler.step(val_loss)

                self.step += 1

                if self.step == self.max_steps:
                    self.save_to_checkpoint()
                    print("Ending training...")
                    exit(0)

    def train_step(self, features: tuple):
        device = next(self.model.parameters()).device
        for param in self.model.parameters():
            param.grad = None

        sample_mix = _map2(_map(features[0])).to(device)
        sample_soi = _map(features[1]).to(device)

        with self.autocast:
            predicted = self.model(sample_mix)
            loss = self.loss_fn(predicted, sample_soi)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.trainer.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    @torch.no_grad()
    def validate(self):
        device = next(self.model.parameters()).device
        self.model.eval()

        loss = 0
        for features in tqdm(
            self.val_dataloader, 
            desc=f"Running validation after step {self.step}"
        ):
            sample_mix = _map2(_map(features[0])).to(device)
            sample_soi = _map(features[1]).to(device)

            with self.autocast:
                predicted = self.model(sample_mix)
                loss += torch.sum(
                    (predicted - sample_soi) ** 2, (0, 1, 2)
                ) / len(self.val_dataset) / np.prod(sample_soi.shape[1:])

        self.writer.add_scalar('val/loss', loss, self.step)
        self.model.train()

        return loss
    

def train(cfg: Config, gen):
    """Training on a single GPU."""
    torch.backends.cudnn.benchmark = True

    model = Wave(cfg.model).cuda()

    learner = WaveLearner(cfg, model, gen)
    learner.restore_from_checkpoint()
    learner.train()
