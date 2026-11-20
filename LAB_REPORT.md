## Use it for your answers, tables and short terminal excerpts. Start with a table of at least six issues. For each one, list the file or setting, explain the problem and suggest a fix.

# AI53 MLOps Project — Phase 1 Lab Report

## Milestone 0: Starter Repository Audit

| # | File / Setting | Problem | Suggested Fix |
|---|---|---|---|


### Engineering Checkpoint 1
**Why is a working `model.pt` insufficient evidence that an experiment is reproducible?**

A standalone `model.pt` file only stores the final trained weights—it contains no record of the exact dataset version, preprocessing parameters, Git code commit, Python/PyTorch package versions, hyperparameters, or random seeds used to produce those weights.