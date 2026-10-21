import json
import tempfile
import unittest
from pathlib import Path

import sentencepiece as spm
import torch

from finetune.prepare_tokenizer import (
    build_tokenizer,
    extract_corpus,
    remap_text_weights,
)


class TokenizerTest(unittest.TestCase):
    def test_row_mapping_preserves_initial_token_and_maps_bias(self):
        original = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        state = {
            "text_emb.weight": original.clone(),
            "depformer_text_emb.weight": original.clone(),
            "text_linear.weight": original[:4].clone(),
            "text_linear.bias": torch.arange(4, dtype=torch.float32),
            "depformer.other": torch.ones(2),
        }
        remap_text_weights(
            state,
            {
                "base_vocab_size": 4,
                "new_vocab_size": 6,
                "rows": [[0], [1], [2], [3], [1, 2], [2, 3]],
            },
        )
        for key in ("text_emb.weight", "depformer_text_emb.weight"):
            torch.testing.assert_close(state[key][:4], original[:4], rtol=0, atol=0)
            torch.testing.assert_close(state[key][6], original[4], rtol=0, atol=0)
            torch.testing.assert_close(state[key][4], original[1:3].mean(0))
        assert state["text_linear.weight"].shape == (6, 3)
        assert state["text_linear.bias"][4].item() == 1.5
        torch.testing.assert_close(state["depformer.other"], torch.ones(2))

    def test_train_eval_collection_and_fresh_spm_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_text = root / "base.txt"
            base_text.write_text("hello world this is a text test\n" * 20)
            spm.SentencePieceTrainer.train(
                input=str(base_text),
                model_prefix=str(root / "base"),
                vocab_size=300,
                hard_vocab_limit=False,
                byte_fallback=True,
                unk_id=0,
                bos_id=1,
                eos_id=2,
                pad_id=3,
                num_threads=1,
            )
            manifests = []
            for split, words in (("train", "한국어 학습"), ("eval", "발음 평가")):
                audio = root / f"{split}.wav"
                audio.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "alignments": [[words, [0, 1], "SPEAKER_MAIN"]],
                        },
                        ensure_ascii=False,
                    )
                )
                manifest = root / f"{split}.jsonl"
                manifest.write_text(
                    json.dumps({"path": str(audio), "duration": 1}) + "\n"
                )
                manifests.append(str(manifest))
            corpus = root / "corpus.txt"
            report = extract_corpus(manifests + manifests, corpus)
            assert report["recordings"] == 2
            assert "평가" in corpus.read_text()
            base = spm.SentencePieceProcessor(model_file=str(root / "base.model"))
            for mode in ("extend", "replace"):
                output = root / mode
                report = build_tokenizer(
                    root / "base.model",
                    manifests,
                    output,
                    mode=mode,
                    vocab_size=300,
                    add_tokens=9,
                )
                prepared = spm.SentencePieceProcessor(
                    model_file=str(output / "tokenizer_spm_ko.model")
                )
                assert (
                    prepared.unk_id(),
                    prepared.bos_id(),
                    prepared.eos_id(),
                    prepared.pad_id(),
                ) == (0, 1, 2, 3)
                assert prepared.decode(prepared.encode("한국어 발음")) == "한국어 발음"
                assert report["prepared_unk_tokens"] == 0
                assert report["prepared_tokens"] < report["base_tokens"]
                if mode == "extend":
                    assert len(prepared) == len(base) + 9
                    for idx in range(len(base)):
                        assert base.id_to_piece(idx) == prepared.id_to_piece(idx)
                rows = json.loads((output / "rows.json").read_text())["rows"]
                assert rows[:4] == [[0], [1], [2], [3]]
                with self.assertRaises(FileExistsError):  # noqa: PT027 - unittest runner
                    build_tokenizer(root / "base.model", manifests, output)


if __name__ == "__main__":
    unittest.main()
