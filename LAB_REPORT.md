## Use it for your answers, tables and short terminal excerpts. Start with a table of at least six issues. For each one, list the file or setting, explain the problem and suggest a fix.

# AI53 MLOps Project — Phase 1 Lab Report

## Milestone 0: Starter Repository Audit

| # | File / Setting | Problem | Suggested Fix |
|---|---|---|---|
| 1 | src/reproducibility.py | Only random is seeded while the DataLoader shuffle generator and worker seeds are not controlled, so two runs so two runs with random_state: 42 can shuffle and initialise weights differently. | Seed every random number generator. |
| 2 | src/validate_data.py (validate_dataset) |  get NotImplementedError | Check splits, classes, counts (1034 / 133 / 128) and decode every image with Pillow, then write a stable report with no timestamps. |
| 3 | .gitignore | The comment says "DVC-managed" but DVC isn't initialised yet. Git ignores the images and nothing else tracks them. | dvc init, dvc add data/raw and set a DVC remote (Milestone) |
| 4 | Missing dvc.yaml | There is no pipeline definition. | Write a dvc.yaml |
| 5 | src/provenance.py (build_run_manifest) | Raises NotImplementedError | Collect these with subprocess
| 6 | src/data.py build_training_loaders | val_loader is built without the seeded generator. | Keep shuffle=False for validation and the test set, and keep worker_init_fn=seed_worker so more workers stay repeatable.

### Engineering Checkpoint 1
**Why is a working `model.pt` insufficient evidence that an experiment is reproducible?**

A standalone `model.pt` file only stores the final trained weights—it contains no record  of the exact dataset version, preprocessing parameters, Git code commit, Python/PyTorch package versions, hyperparameters, or random seeds used to produce those weights.

## Milestone 1: Make Training Repeatable

### Engineering Checkpoint 2
No, the seed make a random choices (weight init, shuffle order, augmentation) the same but the arithmetic can still differ between machines.

### Engineering checkpoint 3
- Git remote store my code, configs and history plus the sage data/raw/raw.dvc pointer file with hash images 
- DVC stores the actual raw images files and it saves under their content hashes
- Git push didnt send the images we need to have dvc push to upload the images in dvc remote 

