<!-- FLAGGED
Response to Major Comment 2 (heterogeneous experiment is weaker than the
wording suggests):

We agree the original Experiment 2 represented quantity skew only, since the
client weight vector [0.10, 0.15, 0.30, 0.25, 0.20] was applied uniformly to
every class, preserving class proportions across clients. We have added a
third federated setting, Experiment 3, using a per-class Dirichlet(alpha=0.3)
partition of the training data (Hsu et al., 2019), a standard method for
simulating genuine label-distribution skew in federated learning. Validation
and test partitions were kept uniform across clients, identical to Experiment
1, isolating label-distribution skew as the only manipulated variable, so
that all evaluation remains on the same pooled test set used elsewhere in the
paper. Total per-client training volume was fixed at 1,500 images (matching
Experiment 1), so quantity is not a confound in this new setting.

We considered a source-aware/domain-skewed client design, as suggested as a
stronger alternative, but did not adopt it: our own Table [tab:dataset_summary]
documents that EMSID incorporates images from MSID, so post-deduplication,
client boundaries defined by nominal source dataset would not correspond to
genuinely independent visual distributions and would risk being flagged as
a mislabeled or misleading heterogeneity axis. We note this explicitly as a
limitation of the source-aware alternative in Section [Limitations].

Under Experiment 3 (Dirichlet label-skew, alpha=0.3), FedProx achieved a
macro-F1 of [XX.XX +/- X.XX] across the five folds, compared with [91.06 +/-
2.20] under Experiment 2 (quantity skew) and [92.47 +/- 2.30] under
Experiment 1 (homogeneous). The paired Wilcoxon test across folds against
Experiment 2 gave p=[X.XX]. These results have been merged into Table 1 as a
new row and Table 5 as a new per-class block, so no new table was required. -->


We agree the original Experiment 2 represented quantity skew only, since the
client weight vector [0.10, 0.15, 0.30, 0.25, 0.20] was applied uniformly to
every class, preserving class proportions across clients. We have added a
third federated setting, Experiment 3, using a per-class Dirichlet(alpha=0.3)
partition of the training data (Hsu et al., 2019), a standard method for
simulating genuine label-distribution skew in federated learning. Validation
and test partitions were kept uniform across clients, identical to Experiment
1, isolating label-distribution skew as the only manipulated variable. Each
client's per-class augmentation target was set proportional to that client's
own Dirichlet-drawn raw class mix (capped at 25x the raw count per class, to
avoid degenerate over-reuse of a handful of source images), rather than a
flat per-class target, so that the induced skew survives augmentation into
the final training set. Per-client training volume is therefore approximately
1,500 images, consistent with Experiment 1, but not manipulated to force
identical class balance across clients.

We considered a source-aware/domain-skewed client design, as suggested as a
stronger alternative, but did not adopt it: our own Table [tab:dataset_summary]
documents that EMSID incorporates images from MSID, so post-deduplication,
client boundaries defined by nominal source dataset would not correspond to
genuinely independent visual distributions. We note this explicitly as a
limitation of the source-aware alternative in Section [Limitations].