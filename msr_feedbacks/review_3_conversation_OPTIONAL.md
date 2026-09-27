## Method and Result changes

**Section: Methodology, insert a new subsubsection after `subsec:fl_training`, right after `\end{algorithm}` and before `\subsection{Evaluation Metrics}`**

Current anchor (last line of that block, unchanged, just showing where to insert after):
```
\end{algorithm}
```

Insert this new subsubsection right after it, before `\subsection{Evaluation Metrics}`:

```latex
\subsubsection{Controlled FedAvg versus FedProx Comparison}
\label{subsec:fedavg_fedprox_controlled}
FedAvg \citep{ref37} and FedProx \citep{ref38} are compared under a controlled protocol so that any performance difference can be attributed to the aggregation strategy rather than to incidental differences in initialization or data ordering. For each fold, both configurations are trained from an identical random seed reset immediately before training begins, so that the backbone, attention, and head initialization and the per-round client batch order are identical between the two runs. The only parameter that differs between the two configurations is the proximal coefficient $\mu$ in Eq.~\eqref{eq:fedprox_local}, set to $\mu=0$ for FedAvg and $\mu=0.01$ for FedProx, with all other hyperparameters, the fold split, augmentation, local epochs, and learning rates held identical (Table~\ref{tab:hyperparams}). Both configurations are evaluated under the quantity-skew heterogeneous setting (Section~\ref{subsec:client_partition}) across all five folds.

Let $f_k^{\mathrm{FedProx}}$ and $f_k^{\mathrm{FedAvg}}$ denote the macro-F1 achieved by FedProx and FedAvg on fold $k$ under this matched protocol. The paired fold-wise comparison is summarized as follows:
\begin{equation}
    \label{eq:paired_diff}
    \Delta_k = f_k^{\mathrm{FedProx}} - f_k^{\mathrm{FedAvg}}, \qquad k=1,\ldots,K,
\end{equation}
\begin{equation}
    \label{eq:cohens_d}
    \bar{\Delta} = \frac{1}{K}\sum_{k=1}^{K}\Delta_k,
    \qquad
    d = \frac{\bar{\Delta}}{s_{\Delta}},
\end{equation}
\begin{equation}
    \label{eq:bootstrap_ci}
    \mathrm{CI}_{95\%} = \left[\widehat{\Delta}^{*}_{(0.025)},\ \widehat{\Delta}^{*}_{(0.975)}\right].
\end{equation}
Here $K=5$ is the number of folds, $\Delta_k$ is the paired macro-F1 difference on fold $k$ (Eq.~\eqref{eq:paired_diff}), $\bar{\Delta}$ is the mean paired difference, $s_{\Delta}$ is the sample standard deviation of $\{\Delta_k\}$, and $d$ is the paired Cohen's effect size (Eq.~\eqref{eq:cohens_d}). $\widehat{\Delta}^{*}$ denotes a bootstrap resample of $\{\Delta_k\}$ drawn with replacement, $B=10{,}000$ such resamples are drawn, and $\mathrm{CI}_{95\%}$ is the resulting 95\% percentile bootstrap confidence interval (Eq.~\eqref{eq:bootstrap_ci}). A supplementary Wilcoxon signed-rank test on $\{f_k^{\mathrm{FedProx}}\}$ versus $\{f_k^{\mathrm{FedAvg}}\}$ is also reported, consistent with its use elsewhere in this study, though the mean difference, effect size, and bootstrap interval are treated as the primary evidence given the small number of folds.
```

**Section: Results and Discussion, insert a new subsection after `subsec:results_main` (right after the paragraph ending `...which is also reflected in Figure~\ref{fig:main_roc}(C,D).`), before `\subsection{Ablation Studies}`**

```latex
\subsection{FedAvg versus FedProx under Controlled Conditions}
\label{subsec:results_fedavg_fedprox}
FedAvg and FedProx are retrained under the matched-seed protocol described in Section~\ref{subsec:fedavg_fedprox_controlled}, so that the comparison isolates the aggregation strategy alone. The resulting rows are reported in Table~\ref{tab:main_cv} and Table~\ref{tab:per_class} alongside the other federated settings. Across the five folds, FedProx achieved a mean paired difference of $TBA$ macro-F1 relative to FedAvg (Cohen's $d = TBA$, 95\% bootstrap CI $[TBA, TBA]$), with a supplementary Wilcoxon signed-rank $p = TBA$. $TBA$ sentence on which classes drove the difference, if any, referencing Table~\ref{tab:per_class}.
```

**Table `tab:main_cv`, add two new rows** (right after the existing `Exp.~2 FL (FedProx), Heterogeneous` row):
```latex
Exp.~2 FL (FedAvg), Heterogeneous
& $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\

Exp.~2 FL (FedProx, matched-seed rerun), Heterogeneous
& $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\
```

**Table `tab:per_class`, add two new 4-row blocks** (same structure as the existing blocks, right after the `Exp.~2 FL (FedProx), Heterogeneous` block):
```latex
\multirow{4}{*}{Exp.~2 FL (FedAvg), Heterogeneous}
& CP & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\
& H  & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\
& M  & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\
& MP & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\
\midrule

\multirow{4}{*}{Exp.~2 FL (FedProx, rerun), Heterogeneous}
& CP & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\
& H  & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\
& M  & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\
& MP & $TBA \pm TBA$ & $TBA \pm TBA$ & $TBA \pm TBA$ \\
```

**Introduction:** now that this experiment genuinely runs, use the FedAvg-vs-FedProx wording we discussed earlier instead of the FedPer wording, since you're doing both now. Let me know once you've decided how to reconcile this with the earlier FedPer contribution-bullet option, since you can't claim both without running FedPer vs FedAvg too, tell me which one you want kept as the bullet and I'll give the final single line.