import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from argparse import ArgumentParser
from omegaconf import OmegaConf
from pathlib import Path

import rfcutils
from src.config_torchwavenet import Config, parse_configs
from src.learner_torchwavenet import train

soi = ['QPSK', 'OFDMQPSK', 'QAM16', 'QPSK2']
interference = ['EMISignal1', 'CommSignal2', 'CommSignal3', 'CommSignal5G1']

def main():
    parser = ArgumentParser(description="Train a Diffwave model.")
    parser.add_argument("soi", type=str, default="QPSK", choices=soi, help="Index for SOI Type.")
    parser.add_argument("interference", type=str, default="CommSignal3", choices=interference, help="Index for Interference Type.")
    parser.add_argument("-id", "--identifier", type=str, default="wavenet", help="Configuration file name for model.")
    args = parser.parse_args()
    
    # Set config
    s = args.soi
    i = args.interference
    m = args.identifier
    cfg = OmegaConf.load(f'src/configs/{m}.yml')
    cfg = Config(**parse_configs(cfg))
    ddir = Path(cfg.data.data_dir)
    cfg.data.data_dir = str(ddir / f"interferenceset_frame/{i}_raw_data.h5")
    cfg.data.val_data_dir = str(ddir / f"testset1_frame/{i}_test1_raw_data.h5")
    cfg.model.input_channels = 2*cfg.data.num_ant
    cfg.model_dir = f"models/{s}_{i}_{m}"
    
    # Setup training
    if m.startswith("wavenet"):
        train(cfg, SOI_Generator(s))
    else:
        raise ValueError(f'Unknown model identifier {m}')

class SOI_Generator:
    def __init__(self, soi_type):
        self.soi_type = soi_type
    def __call__(self, n, s_len):
        if self.soi_type == 'QPSK':
            return rfcutils.generate_qpsk_signal(n, s_len//16)
        elif self.soi_type == 'OFDMQPSK':
            return rfcutils.generate_ofdm_signal(n, s_len//80)
        elif self.soi_type == 'QAM16':
            return rfcutils.generate_qam16_signal(n, s_len//16)
        elif self.soi_type == 'QPSK2':
            return rfcutils.generate_qpsk2_signal(n, s_len//4)
        
if __name__ == "__main__":
    main()
