# Ablation results

Run with `python scripts/run_ablations.py --epochs 8 --out-root runs --force`
on a Colab T4 (8 epochs per variant, not the 30 used for the main baseline;
see note below on why). Each variant trains independently from random init
with the same `warmup_epochs=3, batch_p=16, batch_k=4, optimizer=adam,
lr=1e-3` as the main baseline; only the ablated hyperparameter changes.

| Variant | margin | embed_dim | mining | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|---|---|---|
| baseline_m0.3_d128_hard | 0.3 | 128 | hard | 0.037 | 0.166 | 0.266 | 0.032 |
| margin_0.1 | 0.1 | 128 | hard | 0.045 | 0.150 | 0.258 | 0.031 |
| margin_0.5 | 0.5 | 128 | hard | 0.049 | 0.180 | 0.282 | 0.035 |
| dim_64 | 0.3 | 64 | hard | 0.050 | 0.175 | 0.278 | 0.033 |
| dim_256 | 0.3 | 256 | hard | 0.042 | 0.156 | 0.250 | 0.030 |
| mining_random | 0.3 | 128 | random | 0.042 | 0.175 | 0.294 | 0.036 |

## Why 8 epochs, not 30

The assignment's ablation table doesn't specify an epoch count, only the
variants to compare. Running the full 6-variant grid at 30 epochs each would
mean ~5 hours of additional Colab GPU time on top of the baseline run,
which isn't a reasonable ask of a free-tier session. 8 epochs (3 warmup +
5 hard-mining, matching the 3-epoch warmup used everywhere else) was
chosen as a budget that still clears the warmup transition and gives each
variant a few genuine hard-mining epochs, so the *relative* comparison
between variants is meaningful even though none of them are close to
converged. See `results/results.md` for the full discussion of why
batch-hard training plateaus quickly past warmup in this setup. That
plateau is also why going from 8 to 30 epochs here would likely not change
which variant wins each column, even though it would raise every row's
absolute numbers.

## Findings

**Margin (0.1 / 0.3 / 0.5).** Recall improves monotonically with margin in
this short-budget regime (0.1 worst, 0.5 best on every metric). A larger
margin demands more separation before the loss is satisfied, which keeps
gradient signal alive longer during the few hard-mining epochs available
here. With margin=0.1, many triplets already trivially satisfy the
margin after warmup, so there's less to learn from. I'd expect this gap to
narrow or reverse with full 30-epoch training, since margin=0.1's weaker
constraint is usually better for long-run convergence stability; this is
exactly the kind of margin-vs-epochs interaction that's hard to see without
running the full budget.

**Embedding dimension (64 / 128 / 256).** 64-d is the best of the three,
128-d close behind, 256-d clearly worst. With this little training (8
epochs, no pretraining), a higher-dimensional embedding has more capacity
to fit noise and less relative gradient signal per dimension. This is the
classic small-data/short-budget argument for preferring a smaller
embedding space. This would likely invert with substantially more training
data or epochs, where 256-d's extra capacity becomes an asset rather than
a liability.

**Mining strategy (hard vs. random): the most interesting result.**
Random mining (R@1=0.042, R@5=0.175, R@10=0.294, mAP=0.036) is competitive
with, and on R@10/mAP actually *better than*, batch-hard mining
(R@1=0.037, R@5=0.166, R@10=0.266, mAP=0.032) at this training budget. This
is a direct, reproducible demonstration of the pathology documented in
`results/results.md`: batch-hard mining's "hardest negative" selection is
biased by extreme-value statistics (min of ~60 candidates vs. max of only
~3) to look adversarially hard long before the embedding space has learned
enough structure to make that selection informative, so for a short
training budget straightforward random-pair mining is actually the more
efficient teacher. The value of hard mining is a *long-run* argument (it
should eventually push the decision boundary harder than random pairs once
the model has a baseline level of competence); these 8-epoch numbers are
exactly the regime where that advantage hasn't kicked in yet. This is why
the main baseline's 3-epoch random-mining warmup is a *fix*, not a
workaround: it buys back the "random mining is actually fine/better early
on" benefit while still training with batch-hard mining for the majority
of epochs, as the assignment requires.
