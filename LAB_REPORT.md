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

## Milestone 6 - Predict which stages need to run again

DVC Predictions:

| Change | VALIDATE | PREPARE | TRAIN | EVALUATE |
| :--- | :---: | :---: | :---: | :---: |
| Change `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| Change `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Edit `src/validate_data.py`| RERUN | REUSED | REUSED | REUSED |
| Edit only `README.md` | REUSED | REUSED | REUSED | REUSED |

### Engineering checkpoint 5:
1. Change `train.learning_rate`. VALIDATE has no params and its files did not change, so it is reused. PREPARE only uses `data.image_size`, so it is reused. TRAIN list `train.learning_rate` under `params`, so it reruns. EVALUATE does not list the learning rate, and its inputs (test images, manifests, `selection.run_id`) are the same. The selected Run ID stays fixed, so the saved test resulst still match that model and it is reused.
2. Edit `src/validate_data.py` with the report unchanged. VALIDATE lists `src/validate_date.py` as a dependency, so its hash changes and VALIDATE reruns. It writes the same reports/data_quality.json(no timestamps), so the hash of PREPARE's dependency is the same and PREPARE is reused. TRAIN and EVALUATE depend on PREPARE's outputs, which did not change, so they reused too.
3. Change `data.image_size`. PREPARE uses this setting, so it reruns and makes new processed images and manifests. TRAIN uses the setting and the new data, so it reruns. EVALUATE also uses the setting and the test images, so its saved results no longer match: NEEDS UPDATE. (It would also refuse to run with the old Run ID, because the Run's image size would not match.)

### 6.3 Decisions before training
- **Main validation metric:** validation macro-F1 at the final epoch.
- **Tie-break:** if two Runs are within 0.01 macro-F1, pick the lower final validation loss; if still tied, the shorter training time.
- **Expectation for a lower learning rate (0.0003):** the weights change in smaller steps, so in only 4 epochs the model should learn more slowly. I expect a higher final loss and slightly lower validation scores than Run A, but smoother curves.

### 6.4 Obeservations

| Check | VALIDATE | PREPARE | TRAIN | EVALUATE | Terminal lines |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Baseline `dvc repro train` (first run, Run A) | RERUN | RERUN | RERUN | NOT RUN | `Running stage 'validate'`, `Running stage 'prepare'`, `Running stage 'train'` |
| Run again, no change | REUSED | REUSED | REUSED | NOT RUN | `Stage 'validate' didn't change, skipping ... Data and pipelines are up to date.` |
| Comment added to `src/validate_data.py` | RERUN | REUSED | REUSED | NOT RUN | `Running stage 'validate'`, `Stage 'prepare' didn't change, skipping`, `Stage 'train' didn't change, skipping` |
| `train.learning_rate = 0.0003` (Run B) | REUSED | REUSED | RERUN | NOT RUN | `Stage 'validate' didn't change, skipping`, `Stage 'prepare' didn't change, skipping`, `Running stage 'train'` |

The baseline ran all three stages because the pipeline had never run through DVC before (no dvc.lock yet). The other results matched my predictions.

- Run A ID: < 0ab0151755cf4bc9a7868e8ead6ccf4d >
- Run B ID: < feff4e6f88af47a59caa267a545dd2cb >

## Milestone 7 - Compare three controlled runs
Run C: learning rate back to 0.001, and one change: train.dropout from 0.30 to 0.50.

**Expectation:** more dropout switches off more neurons during training. This reduces overfitting but makes learning harder, so in only 4 epochs I expect a slightly higher training loss and similar or slightly lower validation scores than Run A.


| Run | Controlled change | Val. Accuracy | Val. Macro-F1 | Training time | MLflow Run ID |
| :---: | :--- | ---: | ---: | ---: | :--- |
| A | Baseline (`lr` 0.001, `dropout` 0.30) | 0.639| 0.625 | 10.5 s | `0ab0151755cf4bc9a7868e8ead6ccf4d` |
| B | `learning_rate` 0.0003 | 0.617 | 0.609 | 10.5 s | `feff4e6f88af47a59caa267a545dd2cb` |
| C | `dropout` 0.50 | 0.624| 0.614 | 14.2s s | `f09b85d8c510474d9184595b94ef021e` |

### Engineering Checkpoint 6:
I use validation macro-F1 because it computes the F1 score for each of the three classes and averages them with equal weight. A model that is weak on one disease class gets a lower macro-F1 even if its overall accuracy looks good, and in disease detection each class matters. The test set is used only after choosing the model, because if I used test scores to choose, I would be picking the model that happens to fit the test images best, and the final test score would be too optimistic. The test set must stay unseen so it gives an honest estimate of performance on new images.