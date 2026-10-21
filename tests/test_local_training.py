import unittest

import torch
from torch import nn

from finetune.loss import compute_loss_with_mask
from finetune.trainable import freeze_for_local_training


class LocalTrainingTest(unittest.TestCase):
    def test_joint_gradient_and_frozen_weights(self):
        torch.manual_seed(0)
        model = nn.Module()
        model.text_emb = nn.Embedding(8, 4)
        model.transformer = nn.Linear(4, 4)
        model.out_norm = nn.LayerNorm(4)
        model.text_linear = nn.Linear(4, 8)
        model.emb = nn.Embedding(8, 4)
        model.depformer_in = nn.Linear(4, 4)
        model.depformer = nn.Linear(4, 4)
        model.depformer_text_emb = nn.Embedding(8, 4)
        model.linears = nn.Linear(4, 8)
        freeze_for_local_training(model)
        frozen = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if not p.requires_grad
        }
        assert {n.split(".")[0] for n in frozen} == {
            "emb",
            "depformer_in",
            "depformer",
            "depformer_text_emb",
            "linears",
        }
        target = torch.tensor([[[1, 2, 3]]])
        hidden = model.out_norm(model.transformer(model.text_emb(target)))
        text_logits = model.text_linear(hidden)
        audio_logits = model.linears(model.depformer(model.depformer_in(hidden)))
        mask = torch.ones_like(target, dtype=torch.bool)
        text_loss = compute_loss_with_mask(
            text_logits, target, mask, "text", text_padding_ids=set()
        )
        audio_loss = compute_loss_with_mask(audio_logits, target, mask, "audio")
        parameter = model.transformer.weight
        text_grad = torch.autograd.grad(text_loss, parameter, retain_graph=True)[0]
        audio_grad = torch.autograd.grad(audio_loss, parameter, retain_graph=True)[0]
        assert audio_grad.abs().sum().item() > 0
        optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        (text_loss + audio_loss).backward()
        torch.testing.assert_close(parameter.grad, text_grad + audio_grad)
        optimizer.step()
        assert not torch.equal(before, parameter)
        for name, p in model.named_parameters():
            if name in frozen:
                assert p.grad is None
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
