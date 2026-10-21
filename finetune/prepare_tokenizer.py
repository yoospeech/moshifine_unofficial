"""Train Korean SentencePiece and map its vocabulary onto pretrained Moshi rows.

Also runnable before training with ``python -m finetune.prepare_tokenizer --help``.
No model weights or GPU are needed to build the tokenizer and row mapping.
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

HANGUL = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")


def manifest_paths(sources: str) -> list[Path]:
    paths = []
    for source in sources.split(","):
        if not source.strip():
            continue
        # Same path[:sampling_weight] syntax as the SFT data loader.
        path = Path(source.strip().split(":")[0])
        paths.extend(sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])
    return paths


def extract_corpus(
    sources: list[str], output: Path, max_audio_duration_sec: float | None = None
) -> dict:
    """Read all speakers' alignments, grouping words into utterances.

    Train/eval selection is explicit at the caller. Duplicate audio paths are
    counted once; sampling weights do not change the tokenizer corpus.
    """
    seen = set()
    recordings = lines = 0
    digest = hashlib.sha256()
    with output.open("w", encoding="utf-8") as dst:
        for sources_spec in sources:
            for manifest in manifest_paths(sources_spec):
                for raw in manifest.read_text(encoding="utf-8").splitlines():
                    if not raw.strip():
                        continue
                    record = json.loads(raw)
                    if max_audio_duration_sec is not None:
                        if max_audio_duration_sec <= 0:
                            raise ValueError(
                                "max_audio_duration_sec must be greater than zero"
                            )
                        if float(record["duration"]) >= max_audio_duration_sec:
                            continue
                    audio = Path(record["path"])
                    if audio.resolve() in seen:
                        continue
                    seen.add(audio.resolve())
                    info = json.loads(
                        audio.with_suffix(".json").read_text(encoding="utf-8")
                    )
                    recordings += 1
                    words = []
                    previous_speaker = None
                    previous_end = None

                    def flush():
                        nonlocal lines
                        text = " ".join(words).strip()
                        if text:
                            encoded = (text + "\n").encode("utf-8")
                            dst.write(text + "\n")
                            digest.update(encoded)
                            lines += 1
                        words.clear()

                    for word, (start, end), speaker in info["alignments"]:
                        if words and (
                            speaker != previous_speaker
                            or start - previous_end > 1.0
                            or len(" ".join(words)) + len(word) > 800
                        ):
                            flush()
                        words.append(" ".join(word.split()))
                        previous_speaker, previous_end = speaker, end
                    flush()
    if not lines:
        raise ValueError("No text found in tokenizer training manifests")
    return {
        "recordings": recordings,
        "corpus_lines": lines,
        "corpus_sha256": digest.hexdigest(),
    }


def read_proto(path: Path):
    proto = pb.ModelProto()
    proto.ParseFromString(path.read_bytes())
    return proto


def piece_rows(base, piece: str) -> list[int]:
    idx = base.piece_to_id(piece)
    if base.id_to_piece(idx) == piece:
        return [idx]
    ids = base.encode(piece.replace("▁", " "))
    if not piece.startswith("▁") and ids and base.id_to_piece(ids[0]) == "▁":
        ids = ids[1:]
    return ids or [base.unk_id()]


def token_statistics(corpus: Path, base, prepared) -> dict:
    before = after = unknown = 0
    for line in corpus.read_text(encoding="utf-8").splitlines():
        before += len(base.encode(line))
        ids = prepared.encode(line)
        after += len(ids)
        unknown += ids.count(prepared.unk_id())
    return {
        "base_tokens": before,
        "prepared_tokens": after,
        "token_reduction_fraction": 1 - after / before,
        "prepared_unk_tokens": unknown,
    }


def build_tokenizer(
    base_path: Path,
    sources: list[str],
    output_dir: Path,
    mode: str = "extend",
    vocab_size: int = 10000,
    add_tokens: int = 4096,
    max_audio_duration_sec: float | None = None,
) -> dict:
    if mode not in {"extend", "replace"} or add_tokens <= 0:
        raise ValueError("Invalid tokenizer mode or add_tokens")
    # A new run always starts fresh; never silently reuse mismatched artifacts.
    output_dir.mkdir(parents=True, exist_ok=False)
    base = spm.SentencePieceProcessor(model_file=str(base_path))
    base_proto = read_proto(base_path)
    if base_proto.trainer_spec.model_type != pb.TrainerSpec.UNIGRAM:
        raise ValueError("This vocabulary preparation supports unigram base tokenizers")
    if (base.unk_id(), base.bos_id(), base.eos_id(), base.pad_id()) != (0, 1, 2, 3):
        raise ValueError("Expected Moshi special IDs UNK/BOS/EOS/PAD = 0/1/2/3")
    corpus = output_dir / "corpus.txt"
    report = extract_corpus(sources, corpus, max_audio_duration_sec)
    prefix = output_dir / "korean"
    # This model is learned from scratch, not from the base tokenizer's pieces.
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=1.0,
        hard_vocab_limit=False,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        normalization_rule_name=base_proto.normalizer_spec.name or "nmt_nfkc",
        add_dummy_prefix=base_proto.normalizer_spec.add_dummy_prefix,
        remove_extra_whitespaces=base_proto.normalizer_spec.remove_extra_whitespaces,
        split_digits=True,
        num_threads=1,
        shuffle_input_sentence=False,
        max_sentence_length=16384,
    )
    korean = read_proto(prefix.with_suffix(".model"))
    if mode == "extend":
        existing = {p.piece for p in base_proto.pieces}
        candidates = [
            p
            for p in korean.pieces
            if p.type == pb.ModelProto.SentencePiece.NORMAL
            and p.piece not in existing
            and HANGUL.search(p.piece)
        ]
        candidates.sort(key=lambda p: (-p.score, p.piece))
        chars = [p for p in candidates if len(p.piece) == 1]
        if len(chars) > add_tokens:
            raise ValueError(
                f"add_tokens must cover {len(chars)} observed Hangul characters"
            )
        if len(candidates) < add_tokens:
            raise ValueError(
                f"Only {len(candidates)} Korean pieces; reduce add_tokens or increase vocab_size"
            )
        char_names = {p.piece for p in chars}
        selected = (
            chars
            + [p for p in candidates if p.piece not in char_names][
                : add_tokens - len(chars)
            ]
        )
        final = base_proto
        for piece in selected:
            final.pieces.add().CopyFrom(piece)
    else:
        final = korean
    final.trainer_spec.vocab_size = len(final.pieces)
    final.ClearField("self_test_data")
    tokenizer = output_dir / "tokenizer_spm_ko.model"
    tokenizer.write_bytes(final.SerializeToString())
    prepared = spm.SentencePieceProcessor(model_file=str(tokenizer))
    if mode == "extend":
        assert all(
            base.id_to_piece(i) == prepared.id_to_piece(i) for i in range(len(base))
        )
    # Includes an identity mapping for original/special/byte pieces.
    mapping = {
        "base_vocab_size": len(base),
        "new_vocab_size": len(prepared),
        "rows": [piece_rows(base, p.piece) for p in final.pieces],
    }
    (output_dir / "rows.json").write_text(json.dumps(mapping) + "\n")
    report.update(
        {
            "mode": mode,
            "sources": sources,
            "max_audio_duration_sec": max_audio_duration_sec,
            "base_vocab_size": len(base),
            "new_vocab_size": len(prepared),
            # Keep the training-start log readable while showing actual additions.
            "new_token_examples": [piece.piece for piece in final.pieces[-10:]],
            "base_tokenizer_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            **token_statistics(corpus, base, prepared),
        }
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def remap_text_weights(state: dict, mapping: dict) -> None:
    """Preserve old rows and initialize new pieces from original token means.

    The final embedding row is Moshi's initial token, not a tokenizer piece.
    Relocate it to new_vocab_size, including in the frozen depth embedding.
    """
    old, new = mapping["base_vocab_size"], mapping["new_vocab_size"]
    rows = mapping["rows"]
    if len(rows) != new or any(
        not ids or min(ids) < 0 or max(ids) >= old for ids in rows
    ):
        raise ValueError("Invalid vocabulary row mapping")
    keys = {
        "text_emb.weight": True,
        "depformer_text_emb.weight": True,
        "text_linear.weight": False,
    }
    if "text_linear.bias" in state:
        keys["text_linear.bias"] = False
    for key, has_initial in keys.items():
        weight = state[key]
        if weight.shape[0] != old + int(has_initial):
            raise ValueError(
                f"Unexpected vocabulary shape for {key}: {tuple(weight.shape)}"
            )
        resized = weight.new_empty((new + int(has_initial), *weight.shape[1:]))
        # Vectorize singleton copies; only novel multi-token pieces need means.
        single = [(i, ids[0]) for i, ids in enumerate(rows) if len(ids) == 1]
        if single:
            dst, src = zip(*single)
            resized[list(dst)] = weight[list(src)]
        for i, ids in enumerate(rows):
            if len(ids) > 1:
                resized[i] = weight[ids].float().mean(dim=0).to(weight.dtype)
        if has_initial:
            resized[new] = weight[old]
        state[key] = resized


def configure_checkpoint(checkpoint_info, output_dir: Path):
    """Return new model/tokenizer metadata without copying the multi-GB weights."""
    from dataclasses import replace

    from moshi.models import loaders

    mapping = json.loads((output_dir / "rows.json").read_text())
    config = dict(checkpoint_info.lm_config or loaders._lm_kwargs)
    old = mapping["base_vocab_size"]
    if config.get("text_card") != old or config.get("text_card_out", old) not in (
        None,
        old,
    ):
        raise ValueError("Base model and tokenizer vocabulary sizes do not match")
    config["text_card"] = mapping["new_vocab_size"]
    if "text_card_out" in config:
        config["text_card_out"] = mapping["new_vocab_size"]
    raw = dict(checkpoint_info.raw_config or config)
    raw.update(
        text_card=mapping["new_vocab_size"], tokenizer_name="tokenizer_spm_ko.model"
    )
    if "text_card_out" in raw:
        raw["text_card_out"] = mapping["new_vocab_size"]
    return replace(
        checkpoint_info,
        tokenizer=output_dir / "tokenizer_spm_ko.model",
        lm_config=config,
        raw_config=raw,
    ), mapping


def prepare_for_training(args, checkpoint_info, run_dir: Path):
    import torch.distributed as dist

    if not args.tokenizer.enable:
        return checkpoint_info, None
    output_dir = run_dir / "tokenizer"
    errors = [None]
    if dist.get_rank() == 0:
        try:
            sources = [args.data.train_data]
            if args.tokenizer.include_eval:
                sources.append(args.data.eval_data)
            report = build_tokenizer(
                Path(checkpoint_info.tokenizer),
                sources,
                output_dir,
                mode=args.tokenizer.mode,
                vocab_size=args.tokenizer.vocab_size,
                add_tokens=args.tokenizer.add_tokens,
                max_audio_duration_sec=args.data.max_audio_duration_sec,
            )
            logging.getLogger(__name__).info("Prepared Korean tokenizer: %s", report)
        except Exception as exc:
            errors[0] = f"{type(exc).__name__}: {exc}"
    # Fail every rank together instead of leaving peers waiting at a barrier.
    dist.broadcast_object_list(errors, src=0)
    if errors[0] is not None:
        raise RuntimeError(f"Tokenizer preparation failed: {errors[0]}")
    return configure_checkpoint(checkpoint_info, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["extend", "replace"], default="extend")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--add-tokens", type=int, default=4096)
    parser.add_argument("--max-audio-duration-sec", type=float)
    args = parser.parse_args()
    report = build_tokenizer(
        args.base_tokenizer,
        [args.train_data, args.eval_data],
        args.output_dir,
        args.mode,
        args.vocab_size,
        args.add_tokens,
        args.max_audio_duration_sec,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
