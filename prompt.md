With my current codebase and proposed model, i achieved an internal acc of 0.8525. The logs of training and testing are available in logs folder. You can go through them if needed. 
Issue with the current codebase:
1. overfitting in every experiment which can be seen in the training logs. 
2. train vs val curve's color should be blue => train and red => val => keep in mind. 
3. performance of accuracy is not that much upto the mark as the internal testing accuracy of proposed model is 0.8525. Though the proposed model's performance increased from baseline, why did it decreased in hff only and msda only?

# Your task: 
1. [TOP PRIORITY] Now for your reference I've added a paper and it's codebase in the @/.\reference_materials folder. Go through them properly along with the paper and the codes. They achieved an acc of 96.5% using aptos-19 dataset and we've to beat that anyhow at least by 1% as this one is a recent publication of 2026 from springer nature and i'm targeting that venue too. Go through them properly, analyze them and let me know what are the main things that helped them to achieve the result - also whether we should include them too o get at least a result of 97.5% accuracy. Also if you've got any better or advanced technique than them, you can suggest those too.

2. [MID PRIORITY] One more thing: I've recently go through the GCNet paper and what impressed me the most is in GCNet-S, they were able to squeeze the parallel layers into single one and reduced the parameters from ~20M to ~9.21M during inference keeping the same amount of performance. They probably used RepConv or something for doing so - I'm not sure though. Will we be able to implement this one in our model to feature another novelty? 

3. [MID PRIORITY] In my first planning, I planned to include these ablations:
### Architecture Ablation 
| Experiment | MSDA | HFF | Attention Pool | Aux Head | Ordinal Head |
|------------|------|-----|----------------|----------|--------------|
| Baseline | ✗ | ✗ | ✗ | ✗ | ✗ |
| +MSDA | ✓ | ✗ | ✗ | ✗ | ✗ |
| +HFF | ✗ | ✓ | ✗ | ✗ | ✗ |
| +MSDA+HFF | ✓ | ✓ | ✗ | ✗ | ✗ |
| +AttnPool | ✓ | ✓ | ✓ | ✗ | ✗ |
| +AuxHead | ✓ | ✓ | ✓ | ✓ | ✗ |
| +Ordinal | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Full (Proposed)** | ✓ | ✓ | ✓ | ✓ | ✓ |

### SSL Pretraining Ablation
| Experiment | Pretraining | Val Acc | Val QWK | Test Acc | Test QWK |
|------------|-------------|---------|---------|----------|----------|
| ImageNet pretrain | ImageNet-22k | | | | |
| EyePACS supervised | EyePACS (labeled) | | | | |
| Contrastive only | SSL (InfoNCE) — Option 1 alone | | | | |
| Multi-task only | SSL (lesion tasks) — Option 4 alone | | | | |
| **Full SSL (Proposed)** | SSL (Contrastive + Multi-task) — Combined | | | | |
will i get all the values for these ablation tables from the code-base? If not include the rest. 

4. [Mid Priority] After creating running all the training and testing, there'll be a final script that'll create the necessary tables and other things from the train/output logs for easily updating them into the paper.

5. [TOP Priority] The model should perform really well in external validation too (90%+ is the expected result). So for generalization do necessary things, if needed then FDA (Frequency Domain Adaptation).

NOTE: Only the first fold of the training left are kept in the logs for complexity reduction. msda only and hff only train logs are being deleted but they performed 1-2% less than the baseline.