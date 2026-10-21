# 한국어 토크나이저 준비와 2단계 SFT

## Stage 1 실행

이 프로젝트(`moshi-finetune_t2`)에서 실행한다. 기존 `moshi-finetune_re`의
YAML이 아니라 이 프로젝트에 추가한 YAML을 사용한다.

```bash
cd /home/ysw/Documents/eee/moshi-finetune_t2
/home/ysw/Documents/anaconda3/envs/moshi_finetune/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=1 \
  train.py --config example/moshi_7B_single_gpu_ko.yaml
```

위 Python 환경에서 학습 코드 import와 CPU 테스트를 확인했다.
다른 환경에서는 프로젝트 의존성을 설치한 후 같은 명령의 Python 경로를 바꾼다.
CUDA 실행 가능한 환경이 필요하다. 현재 검증 환경에서는 CUDA가 인식되지 않아
7B GPU 학습과 다중 GPU FSDP 실행은 검증하지 못했다.

설정은 200초 segment, batch 2, 2,000 step이다. 원본 녹음 중 200초 이상인
항목은 train과 eval에서 제외한다. Local 전체 가중치를 직접 학습하므로 LoRA보다
optimizer 메모리 요구량이 크다. 메모리가 부족하면 `duration_sec`를 10으로 줄인다.

## 자동 준비 과정

1. 기본 Moshi checkpoint와 tokenizer 경로를 확인한다.
2. Train과 eval manifest의 모든 녹음에 연결된 JSON `alignments`에서
   모든 화자의 텍스트를 추출한다. 동일 녹음은 중복 수집하지 않는다.
3. 해당 텍스트로 한국어 unigram SentencePiece를 **처음부터 학습**한다.
4. 기본 `extend` 모드에서는 새로운 한국어 토큰 4,096개를 기존 vocabulary에
   추가한다. 기존 토큰 ID와 UNK/BOS/EOS/PAD ID 0/1/2/3은 보존한다.
5. Local text embedding, text 출력층, depth text embedding을 확장한다.
   새 행은 기존 tokenizer로 해당 piece를 표현한 토큰 행들의 평균으로 초기화한다.
   Embedding의 마지막 initial-token 행은 새 vocabulary 뒤로 옮긴다.
6. 확장된 설정으로 모델을 만들고 SFT를 시작한다.

```yaml
tokenizer:
  enable: true
  mode: extend
  vocab_size: 10000
  add_tokens: 4096
  include_eval: true
local_only: true
full_finetuning: false
lora:
  enable: false
save_adapters: false
```

`vocab_size`는 처음부터 학습하는 한국어 후보 tokenizer의 목표 크기이며,
`add_tokens`는 기존 vocabulary에 추가할 한국어 piece 수다.
`mode: replace`는 새 한국어 tokenizer로 vocabulary를 교체한다. 이 경우 일반
토큰 ID가 바뀌고 해당 embedding/output 행도 재배치한다. Stage1 기본값은
기존 ID를 보존하는 `extend`다. 작은 데이터에서 충분한 후보가 나오지 않으면
오류를 내므로 `add_tokens`를 줄이거나 후보 크기를 조정한다.

준비 산출물은 `run_dir/tokenizer/`에 저장한다:

- `corpus.txt`: 학습에 사용한 텍스트
- `korean.model`, `korean.vocab`: 처음부터 학습한 한국어 SentencePiece
- `tokenizer_spm_ko.model`: SFT에 실제 사용하는 tokenizer
- `rows.json`: 기존 모델 행에서 새 모델 행으로의 초기화 매핑
- `report.json`: corpus hash, vocabulary 크기, 토큰 수 비교

Tokenizer 준비 로그와 `report.json`에 새로 반영한 토큰 예시 10개도 표시한다.

모델 전체를 준비 단계에서 별도 복제하지 않고, checkpoint를 읽을 때 텍스트
관련 tensor만 변환한다. 새 run에서는 tokenizer를 새로 학습하며, 기존 run은
`overwrite_run_dir: false`로 보호한다. 재실행은 다른 `run_dir`을 지정한다.

요청대로 eval 텍스트도 tokenizer에 반영한다. `max_audio_duration_sec` 기준도
tokenizer corpus에 동일하게 적용한다. SFT 가중치 업데이트는 train으로만 진행한다.
일반화 평가가 필요하면 tokenizer 학습에도 쓰지 않은 별도 데이터를 사용한다.

## Stage 1 학습 대상

Local은 Moshi의 temporal transformer를 뜻한다. `transformer`, `text_emb`,
`text_linear`, `out_norm`을 학습하고 나머지를 고정한다. Depth transformer,
depth 입력 projection/embedding, audio 출력층, temporal audio embedding이
고정된다. 추가된 depth text embedding 행도 초기화 후 stage1에서는 고정된다.

Loss는 `text CE + audio CE`다. Depth의 파라미터만 `requires_grad=False`이며
forward를 `no_grad`로 감싸거나 hidden state를 detach하지 않는다. Audio CE는
depth를 통과해 temporal hidden state에 gradient를 전달한다. Teacher forcing의
이산적인 텍스트 토큰 ID를 통해서는 미분하지 않는다. 텍스트 embedding 전체가
학습 대상이며, 한국어 행만 선택적으로 학습하는 방식은 아니다.

Checkpoint의 `consolidated/`에 전체 모델, 확장된 `config.json`, 실제 사용한
`tokenizer_spm.model`을 함께 저장한다.

## Stage 2 실행

Stage1이 2,000 step까지 완료된 뒤:

```bash
/home/ysw/Documents/anaconda3/envs/moshi_finetune/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=1 \
  train.py --config example/moshi_7B_ko_stage2.yaml
```

`moshi_7B_ko_stage2.yaml`은 stage1 `checkpoint_002000/consolidated/`의
가중치·config·tokenizer를 사용한다. 다른 step에서 시작하려면 세 경로를 함께
바꾼다. `tokenizer.enable: false`, `local_only: false`, `full_finetuning: true`로
depth를 포함한 전체 모델을 학습한다. Tokenizer를 다시 만들지 않으며,
optimizer와 scheduler는 stage2용으로 새로 시작한다. Stage1과 Stage2 모두
`AdamW8bit`으로 optimizer moment 메모리를 줄이고 BF16 mixed precision을 쓴다.

## 검증

```bash
python -m unittest discover -s tests -v
```

테스트는 token ID 보존, 한국어 tokenizer 신규 학습/교체, 모델 행 초기화,
initial-token 이동, 실제 소형 Moshi의 checkpointing 활성화 상태에서
audio CE → local gradient, frozen weight 불변성, checkpoint 저장/재로딩을 확인한다.

실제 train+eval 1,000개 녹음의 미리보기 결과는
`artifacts/korean_tokenizer_preview/report.json`에 있다. Vocabulary 32,000 →
36,096, 토큰 수 1,310,870 → 236,091(약 82% 감소), UNK 0을 확인했다.
이는 토큰화 효율이며 한국어 생성 품질이나 학습 속도 향상률 측정치는 아니다.
