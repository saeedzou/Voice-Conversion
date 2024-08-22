import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.loggers import TensorBoardLogger
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, AttrDict, build_env

torch.backends.cudnn.benchmark = True

class Hifi_GAN(pl.LightningModule):
    def __init__(self, a, h):
        super(Hifi_GAN, self).__init__()
        self.h = h
        self.a = a
        self.generator = Generator(h)
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()
        self.steps = 0

    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y, _, y_mel = batch # x: mel, y: raw audio, y_mel: mel with different fmax
        y = y.unsqueeze(1) # add channel dimension

        y_g_hat = self.generator(x)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), 
                                      self.h.n_fft, 
                                      self.h.num_mels, 
                                      self.h.sampling_rate, 
                                      self.h.hop_size, 
                                      self.h.win_size, 
                                      self.h.fmin, 
                                      self.h.fmax)
        
        # Discriminator
        if optimizer_idx == 0:
            # Multi-Period Discriminator
            y_dp_hat_r, y_dp_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
            loss_disc_p = discriminator_loss(y_dp_hat_r, y_dp_hat_g)

            # Multi-Scale Discriminator
            y_ds_hat_r, y_ds_hat_g = self.msd(y, y_g_hat.detach())
            loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc = loss_disc_p + loss_disc_s
            return loss_disc
        
        # Generator
        if optimizer_idx == 1:
            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_g_hat_mel, y_mel) * 45 # 45 is lambda value of mel loss according to the paper

            # Feature Matching Loss
            y_dp_hat_r, y_dp_hat_g, fmap_dp_r, fmap_dp_g = self.mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_ds_r, fmap_ds_g = self.msd(y, y_g_hat)
            
            loss_fm_p = feature_loss(fmap_dp_r, fmap_dp_g)
            loss_fm_s = feature_loss(fmap_ds_r, fmap_ds_g)
            
            loss_gen_p, _ = generator_loss(y_dp_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            loss_gen = loss_mel + loss_fm_p + loss_fm_s + loss_gen_p + loss_gen_s
            return loss_gen
        
        def validation_step(self, batch, batch_idx):
            x, y, _, y_mel = batch
            y = y.unsqueeze(1)

            y_g_hat = self.generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), 
                                      self.h.n_fft, 
                                      self.h.num_mels, 
                                      self.h.sampling_rate, 
                                      self.h.hop_size, 
                                      self.h.win_size, 
                                      self.h.fmin, 
                                      self.h.fmax)
            
            val_error = F.l1_loss(y_mel, y_g_hat_mel).item()
            self.log('val_mel_error', val_error)

            if batch_idx <= 4:
                self.logger.experiment.add_figure(f'Validation/y_hat_spec_{batch_idx}', 
                                                  plot_spectrogram(y_g_hat_mel.squeeze(0).cpu().numpy()), 
                                                  self.steps)
                self.logger.experiment.add_audio(f'Validation/y_hat_audio_{batch_idx}', 
                                                 y_g_hat[0], 
                                                 self.steps, 
                                                 sample_rate=self.h.sampling_rate)

        def configure_optimizers(self):
            optim_g = torch.optim.AdamW(self.generator.parameters(), 
                                        lr=self.h.learning_rate, 
                                        betas=[self.h.adam_beta_1, self.h.adam_beta_2])
            optim_d = torch.optim.AdamW(itertools.chain(self.mpd.parameters(), self.msd.parameters()), 
                                        lr=self.h.learning_rate, 
                                        betas=[self.h.adam_beta_1, self.h.adam_beta_2])

            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=self.h.lr_decay)
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=self.h.lr_decay)

            return [optim_d, optim_g], [scheduler_d, scheduler_g]
        
        def train_dataloader(self):
            training_files, _ = get_dataset_filelist(self.a)
            dataset = MelDataset(training_files, 
                                 self.h.segment_size, 
                                 self.h.n_fft, 
                                 self.h.num_mels, 
                                 self.h.hop_size, 
                                 self.h.win_size, 
                                 self.h.sampling_rate, 
                                 self.h.fmin, 
                                 self.h.fmax)
            return DataLoader(dataset, 
                              batch_size=self.h.batch_size, 
                              num_workers=self.h.num_workers,
                              pin_memory=True,
                              drop_last=True)
        
        def val_dataloader(self):
            _, validation_files = get_dataset_filelist(self.a)
            dataset = MelDataset(validation_files, 
                                 self.h.segment_size, 
                                 self.h.n_fft, 
                                 self.h.num_mels, 
                                 self.h.hop_size, 
                                 self.h.win_size, 
                                 self.h.sampling_rate, 
                                 self.h.fmin, 
                                 self.h.fmax)
            return DataLoader(dataset, 
                              batch_size=1, 
                              num_workers=self.h.num_workers,
                              pin_memory=True,
                              drop_last=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    # Setup logging
    logger = pl.TensorBoardLogger(save_dir=a.checkpoint_path, name='logs')
    checkpoint_callback = pl.ModelCheckpoint(dirpath=a.checkpoint_path, save_top_k=-1, every_n_train_steps=a.checkpoint_interval)
    lr_monitor = pl.LearningRateMonitor(logging_interval='step')

    model = Hifi_GAN(h, a)
    
    trainer = pl.Trainer(
        max_epochs=a.training_epochs,
        gpus=h.num_gpus,
        strategy="ddp" if h.num_gpus > 1 else None,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=a.stdout_interval
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()