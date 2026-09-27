Confirmed, Client 1 does have the smallest weight (0.10) in `RUN2_WEIGHTS`, so the reviewer's observation is accurate. Three small edits needed, all wording, no new experiments.

**Section: Results and Discussion, subsection title, line 254**

Current:
```
\subsection{Federated Learning Robustness to Data Poisoning}
```
New:
```
\subsection{Federated Learning Sensitivity to Data Poisoning}
```

**Same subsection, opening line, line 256**

Current:
```
The poisoning experiment examines how the federated models respond when a fraction of Client~1 labels is corrupted. Table~\ref{tab:poisoning} reports the aggregate results.
```
New:
```
The poisoning experiment examines how the federated models respond when a fraction of Client~1, one of the smaller clients under the heterogeneous allocation, has its labels corrupted, so the results should be read as a sensitivity analysis rather than a general robustness guarantee. Table~\ref{tab:poisoning} reports the aggregate results.
```

**Same subsection, closing prose line, line 293**

Current:
```
FedAvg had the highest clean F1 ($0.922\pm0.028$). FedProx remained between $0.910$ and $0.920$ across the poisoned settings, while the trimmed-mean configuration decreased to $0.904$ at $f=0.6$. The pattern was not monotonic across poisoning levels. FedAvg and FedProx follow \citep{ref37,ref38}, while the trimmed-mean aggregation follows \citep{ref61}.
```
New:
```
FedAvg had the highest clean F1 ($0.922\pm0.028$). FedProx remained between $0.910$ and $0.920$ across the poisoned settings, while the trimmed-mean configuration decreased to $0.904$ at $f=0.6$. The pattern was not monotonic across poisoning levels, suggesting that normal training variability across folds is comparable to the measured attack effect. FedAvg and FedProx follow \citep{ref37,ref38}, while the trimmed-mean aggregation follows \citep{ref61}.
```

**Section: Limitations and Future Work**

Your existing line is close but doesn't say Client 1 is a small client and uses "initial assessment" instead of matching the sensitivity analysis language now used in Results.

Current:
```
Robustness was also examined through label-flipping attacks on one federated client, providing an initial assessment of corrupted training labels, although broader poisoning and model-update attacks involving multiple compromised clients were not considered.
```
New:
```
Robustness was also examined through label-flipping attacks on one of the smaller federated clients, providing a sensitivity analysis of corrupted training labels rather than a general robustness guarantee, and broader poisoning and model-update attacks involving multiple compromised clients were not considered.
```

No table, label, or figure reference changes needed, `tab:poisoning` and `subsec:results_poisoning` labels stay exactly as they are so no cross-references break.