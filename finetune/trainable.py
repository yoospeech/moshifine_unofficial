import torch


def freeze_for_local_training(model: torch.nn.Module) -> None:
    """Freeze everything except the temporal transformer and its text branch.

    Only change parameter flags: autograd must still traverse the frozen depth
    branch so that audio CE trains the temporal hidden states.
    """
    trainable_roots = {"transformer", "text_emb", "text_linear", "out_norm"}
    if not hasattr(model, "transformer") or not hasattr(model, "depformer"):
        raise ValueError("local_only expects Moshi temporal and depth transformers.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.split(".", 1)[0] in trainable_roots)
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("No temporal/text parameters found for local_only training.")
