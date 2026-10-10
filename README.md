# Q-Sparse: Scaling Large Language Models with Fully Sparse Activations

This repository provides the official code and training scripts for sparse supervised fine-tuning (SFT) of Qwen2 models with Q-Sparse. It includes environment setup, data preparation, model preparation, and distributed training with DeepSpeed.

Q-Sparse enables fully sparse activations during training by retaining only the largest-magnitude activations according to a configurable sparsity ratio.

- **Oct. 2026:** Code is released!
- **May 2026:** [Q-Sparse](https://openreview.net/pdf?id=MntjMCroiE) is accepted by TMLR 2026.

## Contents

* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Model Preparation](#model-preparation)
* [Sparse Fine-tuning](#sparse-fine-tuning)
* [Training Configuration](#training-configuration)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)

## Installation

We recommend using the NVIDIA PyTorch 23.10 container for training.

```bash
docker run --name nvidia_23_10  --privileged --net=host --ipc=host --gpus=all -v /mnt:/mnt -v /tmp:/tmp -d nvcr.io/nvidia/pytorch:23.10-py3 sleep infinity
docker exec -it nvidia_23_10 bash
```

Clone Q-Sparse under `/mnt` and install the required dependencies:

```bash
cd /mnt
git clone https://github.com/ustcwhy/Q-Sparse.git
cd Q-Sparse
bash scripts/setup.sh
```

The setup script installs the local OpenRLHF package and rebuilds FlashAttention from source:

```bash
cd openrlhf/
pip install .
pip uninstall flash-attn -y
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
```

The training script uses `/your_path` as its storage root. The following commands keep persistent files under `/mnt/q-sparse` while preserving the expected path inside the container:

```bash
mkdir -p /mnt/q-sparse/data
mkdir -p /mnt/q-sparse/checkpoints
mkdir -p /mnt/q-sparse/exp/qwen_sft_sparse_exp
ln -s /mnt/q-sparse /your_path
```

If `/your_path` already exists, either remove the unused path before creating the symbolic link or update the corresponding path variables at the beginning of `scripts/qwen2_sft.sh`.

The expected directory layout is:

```text
/mnt/
├── Q-Sparse/
│   ├── openrlhf/
│   └── scripts/
│       ├── setup.sh
│       └── qwen2_sft.sh
└── q-sparse/
    ├── data/
    │   ├── OpenOrca-hf/
    │   └── MetaMathQA/
    ├── checkpoints/
    │   └── Qwen2-7B/
    └── exp/
        └── qwen_sft_sparse_exp/
```

## Data Preparation

We use [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) and [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) for sparse SFT. By default, the two datasets are sampled with probabilities `0.71` and `0.29`, respectively.

Install the Hugging Face CLI and download the datasets:

```bash
pip install -U huggingface_hub
huggingface-cli download Open-Orca/OpenOrca \
  --repo-type dataset \
  --local-dir /your_path/data/OpenOrca-hf
huggingface-cli download meta-math/MetaMathQA \
  --repo-type dataset \
  --local-dir /your_path/data/MetaMathQA
```

The default data path is defined in `scripts/qwen2_sft.sh`:

```bash
DATA_PATH=/your_path/data/OpenOrca-hf,/your_path/data/MetaMathQA/
```

OpenRLHF recursively loads supported local data files from these directories. The SFT data loader recognizes the following field layouts:

* OpenOrca: `system_prompt`, `question`, and `response`;
* MetaMathQA: `query` and `response`;
* custom instruction data: `input` and `output`;
* custom fields specified with `--input_key` and `--output_key`.

If you change the number of datasets in `DATA_PATH`, update `--dataset_probs` accordingly so that each dataset has one sampling probability.

## Model Preparation

Download the Qwen2 base model to `/your_path/checkpoints`. For example:

```bash
huggingface-cli download Qwen/Qwen2-7B \
  --local-dir /your_path/checkpoints/Qwen2-7B
```

The fourth argument of `scripts/qwen2_sft.sh` specifies the model directory relative to `/your_path/checkpoints`. Therefore, `Qwen2-7B` resolves to:

```text
/your_path/checkpoints/Qwen2-7B
```

## Sparse Fine-tuning

Before launching training, enter the OpenRLHF directory because the script invokes `examples/train_sft.py` using a relative path:

```bash
cd /mnt/Q-Sparse/openrlhf
```

Launch sparse SFT with:

```bash
bash ../scripts/qwen2_sft.sh \
  <EXP_NAME> \
  <LEARNING_RATE> \
  <GLOBAL_BATCH_SIZE> \
  <MODEL_DIR_NAME> \
  <MICRO_BATCH_SIZE> \
  <SPARSE_RATIO> \
  "<EXTRA_ARGS>"
```

For example, the following command fine-tunes Qwen2-7B with a sparsity ratio of `0.5`:

```bash
bash ../scripts/qwen2_sft.sh \
  qwen2-7b-sparse-0.5 \
  2e-5 \
  128 \
  Qwen2-7B \
  1 \
  0.5 \
  "--zero_stage 2"
```

The positional arguments are:

| Argument            | Description                                    | Example               |
| ------------------- | ---------------------------------------------- | --------------------- |
| `EXP_NAME`          | Experiment name and output directory name      | `qwen2-7b-sparse-0.5` |
| `LEARNING_RATE`     | Peak learning rate                             | `2e-5`                |
| `GLOBAL_BATCH_SIZE` | Global training batch size                     | `128`                 |
| `MODEL_DIR_NAME`    | Model directory under `/your_path/checkpoints` | `Qwen2-7B`            |
| `MICRO_BATCH_SIZE`  | Per-GPU micro batch size                       | `1`                   |
| `SPARSE_RATIO`      | Fraction of activations set to zero            | `0.5`                 |
| `EXTRA_ARGS`        | Additional arguments passed to `train_sft.py`  | `"--zero_stage 2"`    |

Pass additional training arguments as a single quoted final argument. For example:

```bash
bash ../scripts/qwen2_sft.sh \
  qwen2-7b-sparse-0.5-z3 \
  2e-5 \
  128 \
  Qwen2-7B \
  1 \
  0.5 \
  "--zero_stage 3 --seed 42"
```

DeepSpeed uses all visible GPUs by default. To restrict training to a subset of GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash ../scripts/qwen2_sft.sh \
  qwen2-7b-sparse-0.5-4gpu \
  2e-5 \
  128 \
  Qwen2-7B \
  1 \
  0.5 \
  "--zero_stage 2"
```

The gradient accumulation factor is computed as:

```text
gradient_accumulation_steps
= GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE / number_of_GPUs
```

Therefore, `GLOBAL_BATCH_SIZE` should be divisible by `MICRO_BATCH_SIZE × number_of_GPUs`.

### Sparse Ratio

For an activation vector, `SPARSE_RATIO=r` retains approximately the largest `(1-r)` fraction of elements by absolute value and masks the rest:

* `r <= 0`: sparsification is disabled;
* `r = 0.5`: approximately 50% of activations are masked;
* larger `r`: a higher degree of activation sparsity.

Use `0 <= r < 1` for sparse fine-tuning.

The trained model and tokenizer are saved to:

```text
/your_path/exp/qwen_sft_sparse_exp/<EXP_NAME>/
```

## Training Configuration

The default configuration in `scripts/qwen2_sft.sh` is:

| Configuration           | Value             |
| ----------------------- | ----------------- |
| Sequence length         | `4096`            |
| Dataset probabilities   | `0.71,0.29`       |
| Maximum samples         | `1,500,000`       |
| Epochs                  | `1`               |
| Precision               | `bfloat16`        |
| FlashAttention          | enabled           |
| Gradient checkpointing  | enabled           |
| Learning-rate scheduler | cosine            |
| Logging interval        | `100` steps       |
| Evaluation              | disabled          |
| Save interval           | `1,000,000` steps |
| Default ZeRO stage      | `2`               |

To modify dataset paths, sampling probabilities, sequence length, number of epochs, or the checkpoint save interval, edit `scripts/qwen2_sft.sh`.

## Acknowledgement

This repository is built on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [Hugging Face Transformers](https://github.com/huggingface/transformers), and [FlashAttention](https://github.com/Dao-AILab/flash-attention). We thank the authors and contributors for making these projects publicly available.

## Citation

If you find Q-Sparse useful in your research, please consider citing our paper:

```bibtex
@article{qsparse,
  author  = {Hongyu Wang and
             Shuming Ma and
             Ruiping Wang and
             Furu Wei},
  title   = {Scaling Large Language Models with Fully Sparse Activations},
  journal = {Trans. Mach. Learn. Res.},
  volume  = {2026},
  year    = {2026},
}
```
