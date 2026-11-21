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

## Milestone 2 - Extract and track the images

### Engineering checkpoint 3
- Git remote store my code, configs and history plus the sage data/raw/raw.dvc pointer file with hash images 
- DVC stores the actual raw images files and it saves under their content hashes
- Git push didnt send the images we need to have dvc push to upload the images in dvc remote 

## Milestone 3 - Implement a Core data-quality gate

During this work, there is an error `cannot identify image file 'data/raw/train/healthy/zz_fake.jpg'`
We need to delete the fake file and run the check again.

## Milestone 4 - Design the DVC pipeline from scratch
dvc dag shows validate -> prepare, then prepare -> train and prepare -> evaluate, with no line between train and evaluate:
```text
        +----------+

        | validate |
        +----------+
              *
         +---------+

         | prepare |
         +---------+
         **        *
+-------+         +----------+

| train |         | evaluate |
+-------+         +----------+
```


  ## Milestone 5 - Record what each MLflow Run used

`src/provenance.py`
build_run_manifest() now returns a dictionary with five parts:

- git: the commit SHA from git rev-parse HEAD, and dirty = true when git status --porcelain lists any file. If Git is not available it writes "unavailable" instead of crashing.
- data: training_data_identity, a hash of the train and validation CSV manifests.
- environment: platform, Python, torch, torchvision, DVC and MLflow versions (from importlib.metadata), and the CUDA runtime when available.
- configuration: image size, batch size, epochs, learning rate, dropout and augmentation settings.
- execution: device, random seed and MLflow Run ID.

`src/train.py`
- write_run_manifest(...) is called right after the Run starts, before any file is written, so the Git status is recorded before training changes the folder.
- training_data_identity is read from the manifest.
- mlflow.log_params(...) logs the 11 required parameters.
- The training manifest is logged as a Dataset input bean-training with target label and context training.
- train_loss, val_loss, val_accuracy and val_macro_f1 are logged with step=epoch inside the loop.
- training_seconds is logged after the loop.
- The curves, summary and manifest are logged as artifacts.
- The model is moved to CPU and put in eval() mode. One validation image (shape 1 x 3 x 96 x 96) is used as the input example, and the signature is inferred from that input and the model output.
- The model is logged as bean-cnn with serialization_format="pickle", code_paths=["src"] and metadata class_names and image_size.

