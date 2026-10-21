"""CPU regression with the real Moshi module, including gradient checkpointing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from moshi.models import loaders

from finetune.checkpointing import Checkpointer
from finetune.loss import compute_loss_with_mask
from finetune.prepare_tokenizer import remap_text_weights
from finetune.trainable import freeze_for_local_training
from finetune.utils import TrainState


class MoshiVocabTest(unittest.TestCase):
    def test_new_token_depth_gradient_and_checkpoint_reload(self):
        config = {
            "delays": [0, 0, 1],
            "n_q": 2,
            "dep_q": 2,
            "card": 16,
            "text_card": 32,
            "dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "hidden_scale": 2,
            "gating": "silu",
            "depformer_gating": "silu",
            "depformer_dim": 16,
            "depformer_num_heads": 2,
            "depformer_num_layers": 1,
            "depformer_dim_feedforward": 32,
            "depformer_weights_per_step": True,
            "gradient_checkpointing": True,
        }
        torch.manual_seed(3)
        base = loaders.get_moshi_lm(
            None, lm_kwargs=config, device="cpu", dtype=torch.float32
        )
        state = base.state_dict()
        remap_text_weights(
            state,
            {
                "base_vocab_size": 32,
                "new_vocab_size": 34,
                "rows": [[i] for i in range(32)] + [[4, 5], [6, 7]],
            },
        )
        config["text_card"] = 34
        model = loaders.get_moshi_lm(
            None, lm_kwargs=config, device="cpu", dtype=torch.float32
        )
        model.load_state_dict(state, strict=True)
        freeze_for_local_training(model)
        frozen = {
            n: p.clone() for n, p in model.named_parameters() if not p.requires_grad
        }
        codes = torch.randint(4, 16, (1, 3, 6))
        codes[:, 0] = 32  # Exercise new rows in both temporal and depth embeddings.
        output = model(codes)
        audio = compute_loss_with_mask(
            output.logits, codes[:, 1:], output.mask, "audio"
        )
        text = compute_loss_with_mask(
            output.text_logits,
            codes[:, :1],
            output.text_mask,
            "text",
            text_padding_ids={0, 3},
        )
        param = next(model.transformer.parameters())
        audio_grad = torch.autograd.grad(audio, param, retain_graph=True)[0]
        assert torch.isfinite(audio_grad).all()
        assert audio_grad.abs().sum() > 0
        optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        (text + audio).backward()
        assert model.text_emb.weight.grad[32].abs().sum() > 0
        optimizer.step()
        for name, p in model.named_parameters():
            if name in frozen:
                assert p.grad is None
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = root / "source.model"
            tokenizer.write_bytes(b"tokenizer-copy-test")
            with patch("finetune.checkpointing.get_rank", return_value=0), patch(
                "finetune.checkpointing.get_world_size", return_value=1
            ), patch("finetune.checkpointing.barrier"):
                saver = Checkpointer(
                    model, TrainState(1, step=1), root, config, tokenizer_path=tokenizer
                )
                saver.save_checkpoint(save_only_lora=False, dtype=torch.float32)
            assert (
                saver.dst_dir / "tokenizer_spm.model"
            ).read_bytes() == tokenizer.read_bytes()
            info = loaders.CheckpointInfo.from_hf_repo(
                hf_repo="unused",
                config_path=saver.dst_dir / "config.json",
                moshi_weights=saver.dst_dir / "consolidated.safetensors",
                tokenizer=saver.dst_dir / "tokenizer_spm.model",
                mimi_weights=Path("unused"),
            )
            reloaded = info.get_moshi(device="cpu", dtype=torch.float32)
            assert reloaded.text_card == 34
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(
                    tensor, reloaded.state_dict()[name], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
