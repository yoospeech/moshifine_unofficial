import logging
import os
from dataclasses import dataclass, field

from simple_parsing.helpers import Serializable

from .data.args import DataArgs


@dataclass
class LoraArgs(Serializable):
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0
    ft_embed: bool = False

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class OptimArgs(Serializable):
    optimizer_type: str = "adamw"  # adamw or adamw8bit
    lr: float = 1e-4
    weight_decay: float = 0.1
    pct_start: float = 0.05

    def __post_init__(self) -> None:
        if self.optimizer_type not in {"adamw", "adamw8bit"}:
            raise ValueError("optim.optimizer_type must be adamw or adamw8bit")


@dataclass
class WandbArgs(Serializable):
    project: str | None = None  # Fill this argument to use wandb.
    offline: bool = False
    key: str | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.project is not None:
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "`wandb` not installed. Either make sure `wandb` is installed or set `wandb:project` to None."
                )

            if len(self.project) == 0:
                raise ValueError("`wandb.project` must not be an empty string.")


@dataclass
class ModelPaths(Serializable):
    hf_repo_id: str | None = "kyutai/moshiko-pytorch-bf16"
    mimi_path: str | None = None
    moshi_path: str | None = None
    tokenizer_path: str | None = None
    config_path: str | None = None

    def __post_init__(self) -> None:
        if self.hf_repo_id is not None and self.config_path is None:
            print(
                "Warning: `hf_repo_id` is set but `config_path` is None. "
                "This will load default models."
            )


@dataclass
class TokenizerArgs(Serializable):
    # Train a fresh Korean SentencePiece model before loading the LM.
    enable: bool = False
    mode: str = "extend"  # extend base vocabulary, or replace with Korean SPM
    vocab_size: int = 10000  # Korean SentencePiece candidate vocabulary
    add_tokens: int = 4096  # Used only in extend mode
    include_eval: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"extend", "replace"}:
            raise ValueError("tokenizer.mode must be extend or replace")
        if self.vocab_size < 300 or self.add_tokens < 1:
            raise ValueError("tokenizer requires vocab_size >= 300 and add_tokens >= 1")


@dataclass
class TrainArgs(Serializable):
    data: DataArgs

    run_dir: str  # Path to the directory where everything will be saved. It needs to be empty.
    # Name of the wandb run, if None it will be set to the name of the run_dir.
    moshi_paths: ModelPaths = field(default_factory=ModelPaths)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    first_codebook_weight_multiplier: float = 1.0
    text_padding_weight: float = 0.5

    optim: OptimArgs = field(default_factory=OptimArgs)
    seed: int = 0
    # Number of steps to accumulate gradients before doing an optimizer step.
    num_microbatches: int = 1

    duration_sec: float = 10
    batch_size: int = 1
    max_norm: float = 1.0  # Gradient clipping.
    max_steps: int = 100  # Number of training steps.
    log_freq: int = 1  # Number of steps between each logging.

    # Number of steps between each checkpoint saving. If inferior to 1, only the last checkpoint will be saved.
    ckpt_freq: int = 0
    save_adapters: bool = True
    # If False, no checkpoints will be saved. This is useful for development.
    do_ckpt: bool = True
    num_ckpt_keep: int | None = 3
    eval_freq: int = 0
    do_eval: bool = False

    # Efficiency
    # Determines whether gradient checkpointing should be utilized or not
    # during the training process. Gradient checkpointing can be beneficial in
    # reducing memory usage at the cost of slightly longer training times.
    gradient_checkpointing: bool = True

    world_size: int | None = field(init=False, default=None)

    # logging
    wandb: WandbArgs = field(default_factory=WandbArgs)

    # LoRA
    lora: LoraArgs | None = field(default_factory=LoraArgs)
    full_finetuning: bool = False
    # Train the temporal/text branch; keep the complete depth/audio branch frozen.
    local_only: bool = False

    param_dtype: str = "bfloat16"

    overwrite_run_dir: bool = False

    def validate_training_mode(self) -> None:
        if self.full_finetuning and self.lora.enable:
            raise ValueError("LoRA must be disabled for full finetuning")
        if not (self.full_finetuning or self.local_only or self.lora.enable):
            raise ValueError("Enable local_only, full_finetuning, or LoRA")

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1

        if self.local_only and (self.full_finetuning or self.lora.enable):
            raise ValueError(
                "local_only requires full_finetuning=false and lora.enable=false. "
                "It updates the temporal/text branch directly, without LoRA."
            )

        if self.tokenizer.enable and not (self.local_only or self.full_finetuning):
            raise ValueError("Tokenizer preparation requires local_only or full_finetuning")

        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1

        if self.lora.enable and not self.save_adapters:
            logging.warning(
                "You have disabled `save_adapters` and are thus merging the "
                "trained LoRA checkpoint into the base model upon checkpointing. "
                "This might lead to OOM errors - make sure you have enough CPU "
                "and GPU memory."
            )
