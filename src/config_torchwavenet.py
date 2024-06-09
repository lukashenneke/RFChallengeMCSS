from dataclasses import MISSING, asdict, dataclass, field
from datetime import datetime
from typing import Optional, List

from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver(
    "datetime", lambda s: f'{s}_{datetime.now().strftime("%H_%M_%S")}')


@dataclass
class ModelConfig:
    input_channels: int = 2
    output_channels: int = 2
    residual_layers: int = 30
    residual_channels: int = 64
    dilation_cycle_length: int = 10


@dataclass
class DataConfig:
    data_dir: str = MISSING
    val_data_dir: str = MISSING
    sig_len: int = 40960
    num_ant: int = 1
    sinr_range: List[int] = field(default_factory=lambda: [-30, 0])
    soi_aoa: Optional[List[float]] = None #[np.pi/4,0] #
    batch_size: int = 16
    num_workers: int = 2


@dataclass
class TrainerConfig:
    learning_rate: float = 2e-4
    max_steps: int = 1000
    max_grad_norm: Optional[float] = None
    fp16: bool = False

    log_every: int = 50
    save_every: int = 2000
    validate_every: int = 100


@dataclass
class Config:
    model_dir: str = MISSING

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig(data_dir="", val_data_dir="")
    trainer: TrainerConfig = TrainerConfig()


def parse_configs(cfg: DictConfig, cli_cfg: Optional[DictConfig] = None) -> DictConfig:
    base_cfg = OmegaConf.structured(Config)
    merged_cfg = OmegaConf.merge(base_cfg, cfg)
    if cli_cfg is not None:
        merged_cfg = OmegaConf.merge(merged_cfg, cli_cfg)
    return merged_cfg


if __name__ == "__main__":
    base_config = OmegaConf.structured(Config)
    config = OmegaConf.load("configs/short_ofdm.yaml")
    config = OmegaConf.merge(base_config, OmegaConf.from_cli(), config)
    config = Config(**config)

    print(asdict(config))
