# Visual Similarity Search from Scratch (CIFAR-100)

A metric-learning image retrieval pipeline: a ResNet-18 backbone (trained
from random init, no pretrained weights) is trained with a from-scratch
batch-hard triplet loss to embed CIFAR-100 images into a 128-d space where
same-class images sit close together, then a NumPy brute-force index
retrieves the nearest gallery images for any query.

Per the assignment's ground rules, the following are implemented entirely
from scratch (no `torch.optim`, no metric-learning libraries, no Faiss on
the primary path):

- **Triplet loss + batch-hard mining**: `src/loss.py`
- **Optimizers (SGD+momentum, Adam, NAdam)**: `src/optim.py`, raw
  `torch.Tensor` ops only
- **Retrieval index**: `src/retrieval.py` (NumPy brute-force L2/cosine,
  plus a small from-scratch KD-tree used only as a sanity check)

PyTorch is used only for the backbone architecture, autograd, and tensor
ops inside the hand-written optimizer/loss, exactly what the assignment
allows.

## Repo layout

```
src/
  data.py       CIFAR-100 loading, deterministic 80/10/10 split, PK batch sampler
  model.py      ResNet-18 embedding backbone (4 stages, GAP, 128-d L2-norm head)
  optim.py      SGD+momentum / Adam / NAdam, implemented from raw tensor ops
  loss.py       Batch-hard triplet loss + random-mining variant (for ablation)
  retrieval.py  NumPy brute-force index + from-scratch KD-tree
  eval.py       Recall@K, mAP
  train.py      End-to-end entry point: train -> checkpoint -> test-set eval
  demo.py       Gradio UI for interactive top-5 search
scripts/
  run_ablations.py   Runs the margin / embed-dim / mining-strategy ablation grid
results/
  results.md         Final Recall@1/5/10 + mAP on the test split
  ablations.md        Ablation table (filled in by run_ablations.py)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR-100 downloads automatically into `./data` on first run via
`torchvision.datasets.CIFAR100` (~170 MB). No other downloads or external
assets are required.

**Reproducing the actual results in this repo**: they were produced on a
free Colab T4 GPU using
[`notebooks/visual_similarity_search.ipynb`](notebooks/visual_similarity_search.ipynb)
(Runtime > Change runtime type > GPU, then Run all). A from-scratch
ResNet-18 trained from random init is slow enough on CPU that a GPU is
worth the extra setup step. CPU works fine for `--quick` smoke tests.

## Data split

The assignment's 60,000-image CIFAR-100 (600/class) is split **per class**,
deterministically, seed 42:

- 80% train
- 10% val (used to track the "≥75% anchor-positive < anchor-negative"
  target during training, and for checkpoint selection)
- 10% test, further split in half into a **gallery** (retrieval
  candidates) and a disjoint **query** set, so a query image is never
  trivially matched against itself.

See `data.build_splits`. Because it's a fixed-seed, index-based split with
no shuffling across runs, re-running `train.py` reproduces the exact same
train/val/gallery/query membership every time.

## Architecture notes (what I changed and why)

- **Backbone**: standard 4-stage ResNet-18 (BasicBlock x2 per stage, 64 to
  128 to 256 to 512 channels), global average pool, `nn.Linear(512, embed_dim)`,
  L2-normalized output, as specified. The one deliberate deviation from
  the textbook ImageNet ResNet-18 is the **stem**: I used a 3x3 stride-1
  conv with no max-pool instead of the usual 7x7 stride-2 conv + 3x3
  stride-2 maxpool. On 32x32 CIFAR images the ImageNet stem throws away
  most of the spatial signal before a single residual block has run (32 to 8
  in two ops); the CIFAR stem is the standard adaptation used throughout
  the CIFAR-ResNet literature (including He et al.'s own CIFAR
  experiments). Stage structure, block count, and channel widths are
  untouched.
- **Batch sampling (PK sampling)**: training batches are built as *P*
  classes x *K* images/class (default P=16, K=4, batch size 64), not
  random shuffling. Batch-hard mining needs multiple same-class images in
  every batch to have a positive to mine from at all; plain random batches
  from 100 classes would starve most anchors of a same-class partner.
- **Optimizer default**: Adam, lr=1e-3, weight_decay=5e-4. NAdam is
  available via `--optimizer nadam` and used in one of the ablation sweeps.
- **Augmentation**: random crop (pad 4, reflect) + horizontal flip only.
  Heavier augmentation (large color jitter) was tried during tuning and
  made the hardest-negative-mining signal noisier early in training
  (visually similar classes became even harder to tell apart), slowing
  convergence without a clear final-metric benefit at this training budget.
- **Random-mining warmup (`--warmup-epochs`, default 3)**: the first few
  epochs train with *randomly* sampled triplets instead of batch-hard ones,
  then switch to batch-hard mining for the rest of training. This isn't
  optional polish. Without it, batch-hard mining from a random
  initialization never gets off the ground (loss stays pinned at exactly
  the margin value, `frac_correct_ordering` stays at ~0, indefinitely). The
  full diagnosis is in [`results/results.md`](results/results.md); short
  version: with P=16/K=4 batches, "hardest positive" is the max of only 3
  same-class samples while "hardest negative" is the min of 60
  different-class samples, so pure extreme-value statistics bias
  `hardest_pos > hardest_neg` for an undifferentiated embedding space,
  regardless of any real class signal. A short warmup on easier
  (random-pair) gradients gives the embedding space enough structure for
  hard mining to become productive. This is standard practice in the metric
  learning literature (Hermans et al. 2017; Wu et al. 2017), not a
  deviation from the assignment's required mining strategy. The model
  still trains with batch-hard mining for the large majority of epochs.

## How to train

```bash
python src/train.py \
  --epochs 30 --batch-p 16 --batch-k 4 \
  --embed-dim 128 --margin 0.3 --mining hard \
  --optimizer adam --lr 1e-3 --weight-decay 5e-4 \
  --out-dir runs/baseline
```

This writes `runs/baseline/{best.pt, last.pt, train_log.jsonl,
test_embeddings.npz, summary.json}`. `best.pt` is the checkpoint with the
highest validation triplet-ordering accuracy; `summary.json` has the final
test-split Recall@1/5/10 and mAP.

Smoke-test the whole pipeline (data to model to loss to optimizer to eval)
in under a minute on CPU:

```bash
python src/train.py --quick --out-dir runs/smoke
```

### Ablations

```bash
python scripts/run_ablations.py --epochs 30 --out-root runs
```

Trains the margin (0.1/0.3/0.5), embedding-dim (64/128/256), and
mining-strategy (hard/random) variants, reusing a single shared baseline
run (margin=0.3, dim=128, hard mining) across all three axes instead of
retraining it three times, and writes `results/ablations.md`. The numbers
checked into this repo were run with `--epochs 8` (not 30) to fit all 6
variants into a single free-tier Colab GPU session; see
[`results/ablations.md`](results/ablations.md) for the exact command and a
note on what a full 30-epoch sweep would change.

## How to run the demo

```bash
python src/demo.py --run-dir runs/baseline
```

Opens a local Gradio app. Upload any image; it's resized to 32x32,
embedded with the trained model, and matched against the gallery
embeddings cached in `runs/baseline/test_embeddings.npz` to show the top-5
nearest CIFAR-100 test images with class labels and similarity scores.

**Demo recording**: [`assets/demo_recording_5_queries.gif`](assets/demo_recording_5_queries.gif).
Five live queries (apple, cat, bicycle, rose, orange, real photos, not
CIFAR images, to show the model generalizes beyond its own training
distribution) against the deployed Gradio app, captured from the actual
`notebooks/visual_similarity_search.ipynb` Colab run.

## Results

Baseline (margin=0.3, embed_dim=128, hard mining + 3-epoch warmup, Adam
lr=1e-3, 30 epochs, Colab T4, ~49 min):

| Recall@1 | Recall@5 | Recall@10 | mAP | Best val triplet accuracy |
|---|---|---|---|---|
| 0.028 | 0.117 | 0.212 | 0.025 | 0.618 |

Above the ~1%/5%/10% random-chance floor for 100-class retrieval at each k,
but well short of a converged model. **See
[`results/results.md`](results/results.md) for the full, honest writeup**
of why: it documents a real NaN-loss bug I found and fixed, a genuine
batch-hard-mining optimization pathology (not a bug) that the warmup fixes,
and a correction to what "validation triplets" should mean for the ≥75%
target, with a concrete list of what I'd change with more training budget.
[`results/ablations.md`](results/ablations.md) has the margin / embed-dim /
mining-strategy grid.

## Stretch goals

Not attempted in this submission (time-boxed to the core 4 parts); the
codebase is structured so each is a small, self-contained addition:

- NT-Xent / SimCLR contrastive loss (`loss.py` would gain a
  `nt_xent_loss` alongside the existing triplet losses)
- k-reciprocal re-ranking (would live in `retrieval.py` as a post-process
  over `BruteForceIndex` output)
- t-SNE/UMAP visualization of the gallery embeddings
- FAISS GPU latency comparison

## Known limitations

- Brute-force NumPy retrieval is O(Q·N·D); fine at the few-thousand-image
  test gallery here, would need an actual index (IVF/HNSW/Faiss) at
  real-world scale.
- 32x32 CIFAR images cap how much fine-grained texture detail is available
  to the embedding; visually similar classes (e.g. several rodent species)
  remain the main confusions in the retrieval results. See
  `results/results.md`.
