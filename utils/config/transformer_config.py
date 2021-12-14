import torch
import dataclasses


@dataclasses.dataclass
class TransformerConfig:
    flip_prob: float = 0.3
