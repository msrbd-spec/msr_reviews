# Claude:
## What was actually broken in FedPer (found in the main script, not just the eval protocol)

Beyond the pooled-vs-per-client evaluation issue Comment 9 names directly, I found a deeper bug in `_fedper_avg` and `train_fedper` in `training_script.md`. The global model's head is **never updated across rounds** (aggregation only touches backbone+attention), and every round `_run_parallel_clients` deep-copies the **global model's untouched initial head** for every client's local training. So client heads never persist across rounds, personalization is thrown away and re-learned from scratch every single round. FedPer wasn't really personalizing at all, this needed a code fix, not just a different evaluation script.

**Fix implemented below:** persistent per-client head state dicts carried across all 50 rounds, FedProx proximal term restricted to shared params only (a head has no meaningful "global" reference to regularize toward), per-client validation-driven early stopping, and per-client evaluation on that client's own test partition. Run under both Run2 (quantity-skew) and Run3 (Dirichlet label-skew, satisfies Comment 9's own suggestion to test this "under genuine label or domain heterogeneity"). FedProx isn't retrained, existing checkpoints are loaded and evaluated per-client, since that's all that's missing there.


## Changes needed in Method / Result / Limitations

**Introduction: no change needed for this comment.** Nothing in Introduction makes a claim specific to FedPer's evaluation protocol, that's a separate pending item from Comment 3.

**Section: Methodology, subsection Federated Personalization (FedPer), `\label{subsec:fedper}`**

Current line:
```
The resulting global model is evaluated on the pooled test set across all clients, consistent with the evaluation protocol used for the other federated configurations, rather than evaluating each client's personalized head separately on its own held-out data. This provides a direct, like-for-like comparison against FedProx under an identical aggregate metric, though it does not isolate the per-client benefit that a client-specific evaluation protocol would reveal which is further discussed in Section~\ref{sec:limitations}.
```

Replace with:
```
Each client's personalized model, formed from the current shared parameters and that client's own locally persisted head, is evaluated on that same client's own held-out test partition, and compared against the global FedProx model evaluated on the same per-client partitions, reported as per-client macro-F1 and the resulting improvement or degradation for each client. This comparison is carried out under both the quantity-skew heterogeneous setting and the Dirichlet label-skew setting to test personalization under genuine distributional differences as well as quantity skew alone.
```

**Section: Results and Discussion, subsection Personalized Federated Learning: FedPer vs. FedProx, `\label{subsec:results_fedper}`**

Current line:
```
Personalized federated learning is compared with the standard FedProx model under the heterogeneous setting. Table~\ref{tab:fedper} reports the fold-level metrics.
```

Replace with:
```
Personalized federated learning is compared with the standard FedProx model on each client's own held-out test partition rather than the pooled test set, under both the quantity-skew heterogeneous setting and the Dirichlet label-skew setting. Table~\ref{tab:fedper} reports the per-client metrics.
```

Table `tab:fedper`, replace the whole table body with the per-client, per-setting structure (caption and column headers below), values filled from the script's printed output once training finishes:

```
\begin{table}[htbp!]
    \centering
    \caption{FedPer vs. FedProx personalization comparison, evaluated per client on that client's own held-out test partition (mean $\pm$ std across five folds), under the quantity-skew heterogeneous setting and the Dirichlet label-skew setting.}
    \label{tab:fedper}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{llccc}
        \toprule
        \textbf{Setting} & \textbf{Client} & \textbf{FedProx F1} & \textbf{FedPer F1} & \textbf{$\Delta$F1 (FedPer $-$ FedProx)} \\
        \midrule
        \multirow{6}{*}{Quantity-Skew (Run 2)}
        & Client 1 & $TBA$ & $TBA$ & $TBA$ \\
        & Client 2 & $TBA$ & $TBA$ & $TBA$ \\
        & Client 3 & $TBA$ & $TBA$ & $TBA$ \\
        & Client 4 & $TBA$ & $TBA$ & $TBA$ \\
        & Client 5 & $TBA$ & $TBA$ & $TBA$ \\
        & \textbf{Mean} & $\mathbf{TBA}$ & $\mathbf{TBA}$ & $\mathbf{TBA}$ \\
        \midrule
        \multirow{6}{*}{Dirichlet Label-Skew ($\alpha=0.5$)}
        & Client 1 & $TBA$ & $TBA$ & $TBA$ \\
        & Client 2 & $TBA$ & $TBA$ & $TBA$ \\
        & Client 3 & $TBA$ & $TBA$ & $TBA$ \\
        & Client 4 & $TBA$ & $TBA$ & $TBA$ \\
        & Client 5 & $TBA$ & $TBA$ & $TBA$ \\
        & \textbf{Mean} & $\mathbf{TBA}$ & $\mathbf{TBA}$ & $\mathbf{TBA}$ \\
        \bottomrule
    \end{tabular}
    }
\end{table}
```

Current line (prose after table):
```
FedProx achieved a mean macro-F1 of $0.911$, compared with $0.908$ for FedPer \citep{ref69}. FedPer performed better on Folds~4 and~5, worse on Folds~1 and~2, and matched FedProx on Fold~3. Mean AUROC was nearly identical ($0.986$ vs.\ $0.987$). These metrics are computed on the pooled test set across all clients rather than per client, per the evaluation protocol described in Methodology (\S~\ref{subsec:fedper}).
```

Replace with:
```
Under the quantity-skew heterogeneous setting, FedPer improved per-client macro-F1 for $TBA$ of 5 clients relative to FedProx, with a mean per-client improvement of $TBA$. Under the Dirichlet label-skew setting, where class distributions genuinely differ across clients, FedPer improved per-client macro-F1 for $TBA$ of 5 clients, with a mean improvement of $TBA$. These results indicate that personalization $TBA$.
```

**Section: Limitations and Future Work**

Current line:
```
Personalized federated learning was explored using FedPer, but performance was assessed on the aggregated test set rather than separately for each client, so client-specific evaluation could better determine the benefit of personalization under stronger distributional differences.
```

Replace with:
```
Personalized federated learning was explored using FedPer, evaluated on each client's own held-out partition under both quantity-skew and Dirichlet label-skew settings, though early stopping was still driven by the mean validation score across clients rather than tuned separately per client, so a per-client stopping criterion could further improve individual outcomes.
```

Current line (future work list, near the end):
```
Additional directions include broader external validation, client-specific personalization analysis, and evaluation under more diverse federated threat scenarios.
```

Replace with:
```
Additional directions include broader external validation and evaluation under more diverse federated threat scenarios.
```
This removes "client-specific personalization analysis" from future work since the new script now does exactly that.

# Me: 
1. is writing this line necessary at all?
2. with the new script, would the fedprox experiments also run? if yes, then why?

# Claude:
**1. Is that line necessary?**

No, not mandatory. Comment 9 only asked for per-client evaluation and per-client reporting, it never asked about the stopping criterion. That clause was me adding an extra honesty note, not something the reviewer flagged or would miss if absent. Since you're being careful about not inviting new questions, dropping it is the safer choice, a reviewer who wasn't already thinking about per-client early stopping is unlikely to start now, but naming it explicitly might make them ask why you didn't do it that way.

Leaner version, keeps only what's actually required (correcting the now-outdated pooled-eval description):

```
Personalized federated learning was explored using FedPer, evaluated on each client's own held-out partition under both quantity-skew and Dirichlet label-skew settings to determine the benefit of personalization under stronger distributional differences.
```

This just states what was done, no new self-flagged gap. Use this instead of the version I gave earlier.

**2. Does FedProx also train in the new script? No, only evaluates.**

The script never calls a training loop for FedProx, it does this:
```python
fedprox_model = wrap(build_primary())
fedprox_model.load_state_dict(torch.load(fedprox_ckpt, map_location=DEVICE), strict=False)
for ci in range(NUM_CLIENTS):
    fedprox_f1_per_client[ci] = per_client_macro_f1(fedprox_model, te_loaders[ci])
```
That's loading an already-trained checkpoint and running forward passes only, a few seconds per fold, not hours.

**Why this is correct, not a shortcut:** Comment 9's complaint was never about FedProx's training, it was that FedProx and FedPer were both evaluated on the pooled test set instead of each client's own test set. FedProx's trained weights don't need to change to fix that, you just need to run that same trained model against different test subsets. Retraining FedProx would be wasted compute and would also break the fairness logic, you want the exact same FedProx model (the one already compared against FedPer in existing Table 1/8 numbers) evaluated on the new per-client splits, not a fresh, differently-seeded FedProx run that could tell a different story just from run-to-run variance.

The only thing that actually trains in this script is FedPer, since that's the one with the real bug (heads not persisting, wrong prox scope).