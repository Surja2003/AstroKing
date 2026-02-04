# ML (Hand/Palm) Training Workspace

This folder is intentionally **separate** from the FastAPI backend.

Goal:
- Pretrain a small CNN on your hand/palm dataset (e.g., gender / age labels).
- Export an **embedding model** (feature extractor) you can reuse for higher-accuracy downstream predictions.
- Export **TFLite** for mobile / lightweight inference.

## 0) Create a Python env (recommended)

Important:
- Use a **separate venv inside `ml/`**.
- On Windows, TensorFlow typically won’t install/import on newer Python versions (e.g. 3.12+). This project expects **Python 3.11** for ML.

From repo root:

```powershell
cd ml
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Start (No Dataset)

This is the fastest way to sanity-check your models without preparing a dataset.

1) Put a test image here (local-only, ignored by git):
- `ml/sample_images/palm1.jpg`

2) Run the guided one-image runner:

```powershell
cd ml
powershell -ExecutionPolicy Bypass -File .\run_one_windows.ps1
```

The runner auto-detects `predict` vs `embed` from the selected `.keras` model.
To force a mode manually:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_one_windows.ps1 -Mode predict
```

It will:
- Ensure `ml/.venv` exists (Python 3.11)
- Install `ml/requirements.txt` (unless you pass `-SkipInstall`)
- Prompt for `predict` vs `embed`, model path, and image path

## Using the `archive/Hands` dataset (front/back filtering)

The dataset under `C:\Users\dasne\Downloads\archive\Hands\Hands` contains both *palmar* (front/palm) and *dorsal* (back of hand) images.
The file `C:\Users\dasne\Downloads\archive\HandInfo.csv` provides metadata including `aspectOfHand` (e.g. `palmar left`, `dorsal right`) and `gender`.

### Extract PALMAR-only images for training

This produces a clean, consistent-view dataset layout:

- `out_dir/MALE/*.jpg`
- `out_dir/FEMALE/*.jpg`

From the `ml/` folder:

- `.venv\Scripts\python.exe extract_handinfo_subset.py --handinfo_csv "C:\Users\dasne\Downloads\archive\HandInfo.csv" --images_dir "C:\Users\dasne\Downloads\archive\Hands\Hands" --out_dir "C:\Users\dasne\Downloads\hands_palmar_gender" --label_source gender --palmar`

Then split and train:

- `.venv\Scripts\python.exe split_dataset.py --data_dir "C:\Users\dasne\Downloads\hands_palmar_gender" --out_dir "C:\Users\dasne\Downloads\hands_palmar_gender_split" --classes "MALE,FEMALE" --train 0.7 --val 0.15 --test 0.15 --seed 42 --hardlink`

If you see the model collapse into predicting only one class, enable imbalance handling:

- `--class_weight balanced` (adds per-sample weights during training and uses weighted validation for checkpoint selection)
- `--fit_verbose 2` (keeps logs readable)

Recommended command (works well on the palmar-only split):

- `.venv\Scripts\python.exe train_cnn.py --mode directory --data "C:\Users\dasne\Downloads\hands_palmar_gender_split" --epochs 6 --fine_tune --fine_tune_epochs 6 --unfreeze_last 40 --fine_tune_lr 1e-5 --class_weight balanced --fit_verbose 2 --out_dir "C:\Users\dasne\Downloads\models_palmar_balanced"`

### (Optional) Train a PALMAR-vs-DORSAL view classifier

Useful if you later want to auto-filter camera frames by view.

- `.venv\Scripts\python.exe extract_handinfo_subset.py --handinfo_csv "C:\Users\dasne\Downloads\archive\HandInfo.csv" --images_dir "C:\Users\dasne\Downloads\archive\Hands\Hands" --out_dir "C:\Users\dasne\Downloads\hands_view" --label_source aspect`

## 1) Dataset layouts supported

### Option A: Folder-per-class (easy)

For gender classification, arrange like:

```
data/
  male/
  female/
```

You can also have more than 2 classes.

### Option B: CSV labels (age regression or multi-task)

If your dataset comes with a CSV, you can use:
- `--mode csv`
- `--csv path/to/labels.csv`
- columns like `path`, `gender`, `age`

(See script help for exact flags.)

## 1.0) Convert CSV → folder-per-class

If you have a labels file like:

```
filename,gender
img1.jpg,male
img2.jpg,female
```

You can convert it into:

```
prepared/
  male/
  female/
```

Command:

```powershell
cd ml
.\.venv\Scripts\python.exe prepare_csv.py --csv "C:\dataset\labels.csv" --images "C:\dataset\images" --out "C:\dataset\prepared" --label_col gender --file_col filename --skip_missing
```

Then run `split_dataset.py` → `train_cnn.py` → `export_tflite.py` as usual.

## 1.1) Quick split script (train/val/test)

If your raw dataset is:

```
C:\Users\dasne\Downloads\archive\
  male\
  female\
```

Create a proper split once:

```powershell
cd ml
.\.venv\Scripts\python.exe split_dataset.py --source "C:\Users\dasne\Downloads\archive" --target "C:\Users\dasne\Downloads\palm_gender_dataset" --classes male,female --train 0.7 --val 0.15 --test 0.15
```

This produces:

```
palm_gender_dataset/
  train/male
  train/female
  val/male
  val/female
  test/male
  test/female
```

## 1.2) Sanity-check the split

```powershell
cd ml
.\.venv\Scripts\python.exe dataset_report.py --root "C:\Users\dasne\Downloads\palm_gender_dataset" --classes male,female
```

Optional duplicate detection across splits (slower):

```powershell
.\.venv\Scripts\python.exe dataset_report.py --root "C:\Users\dasne\Downloads\palm_gender_dataset" --classes male,female --hash
```

## 1.3) One-command Windows runner

This will create the venv (if needed), split the dataset, train, and export TFLite:

```powershell
cd ml
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
```

To skip the split step (if you already ran it):

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -SkipSplit
```

## 2) Train the CNN

### Folder-based classification

```powershell
.\.venv\Scripts\python.exe train_cnn.py --mode directory --data "C:\Users\dasne\Downloads\palm_gender_dataset" --epochs 10
```

Outputs:
- `ml/models/hand_cnn.keras` (full classifier)
- `ml/models/hand_embedding.keras` (embedding model)

## 2.1) Evaluate a saved model (no retraining)

Use this to compare runs (baseline vs fine-tune, etc.) without training again.

If `--data` is a split root containing `train/`, `val/`, `test/`, then `eval_model.py` will evaluate on `test/` by default.

```powershell
cd ml
.\.venv\Scripts\python.exe eval_model.py --model "C:\Users\dasne\Downloads\models_palmar_balanced_weightedval\hand_cnn.keras" --data "C:\Users\dasne\Downloads\hands_palmar_gender_split"
```

To force a specific subset:

```powershell
.\.venv\Scripts\python.exe eval_model.py --model "models\hand_cnn.keras" --data "C:\Users\dasne\Downloads\hands_palmar_gender_split" --subset val
```

## 2.2) Compare multiple models (no retraining)

This prints a compact table (sorted by macro F1), which makes experimentation easy.

```powershell
cd ml
.\.venv\Scripts\python.exe compare_models.py --models \
  "C:\Users\dasne\Desktop\AstroKing\ml\models\hand_cnn.keras" \
  "C:\Users\dasne\Downloads\models_palmar_balanced_weightedval\hand_cnn.keras" \
  --data "C:\Users\dasne\Downloads\hands_palmar_gender_split"
```

### CSV-based (gender + age)

```powershell
.\.venv\Scripts\python.exe train_cnn.py --mode csv --csv ..\data\labels.csv --image_root ..\data --epochs 10
```

## 3) Export to TFLite

```powershell
.\.venv\Scripts\python.exe export_tflite.py --keras_model models\hand_embedding.keras --out models\hand_embedding_float16.tflite --quant float16
```

## 4) Extract embeddings for a downstream model

## 4.0) Quick one-image test (recommended)

Use this when you just want to sanity-check a model on a single image.

### Predict a class (use the classifier model)

```powershell
cd ml
.\.venv\Scripts\python.exe infer_one.py --mode predict --model models\hand_cnn.keras --image path\to\image.jpg
```

### Extract an embedding vector (use the embedding model)

```powershell
cd ml
.\.venv\Scripts\python.exe infer_one.py --mode embed --model models\hand_embedding.keras --image path\to\image.jpg
```

### Machine-readable output (for scripting)

Add `--json` to emit JSON (useful for piping into other scripts):

```powershell
.\.venv\Scripts\python.exe infer_one.py --mode embed --model models\hand_embedding.keras --image path\to\image.jpg --json
```

Exit codes:
- `0` success
- `2` bad/missing image
- `3` model/TensorFlow load or inference failure

```powershell
.\.venv\Scripts\python.exe extract_embeddings.py --model models\hand_embedding.keras --mode directory --data_dir ..\data --out models\embeddings.npz
```

This produces:
- `X`: float32 array of shape `[N, embedding_dim]`
- `y`: labels (if available in directory/CSV)
- `paths`: original image paths

## 5) Train a downstream model (example)

```powershell
.\.venv\Scripts\python.exe train_downstream.py --embeddings models\embeddings.npz --out models\downstream.pkl
```

## 6) Build a small “personality archetype” index (for backend retrieval)

If you don’t have supervised “personality labels” yet, you can still use your CNN embeddings by doing nearest-neighbor matching against a small curated set of archetypes.

Create a folder like:

- `archetypes/`
  - `The Strategist/` (images)
  - `The Empath/` (images)
  - `The Builder/` (images)

Optionally add a JSON metadata file mapping folder name to traits/summary:

```json
{
  "The Strategist": {"summary": "Calm, analytical, plan-first", "traits": ["Analytical", "Calm", "Disciplined"]}
}
```

Build an index `.npz`:

```powershell
.\.venv\Scripts\python.exe build_trait_index.py --model models\hand_embedding.keras --data_dir ..\archetypes --meta ..\archetypes_meta.json --out ..\backend\palm_trait_index.npz
```

The backend will auto-load `backend/palm_trait_index.npz` and expose:
- `POST /scan-palm/personality` (multipart) for ML matches only
- `POST /scan-palm` and `POST /upload-palm` now include a `personality` field

## Notes / safety

- Predicting attributes like age/gender can be sensitive; use with explicit consent and be careful with claims.
- For best results, segment/crop the palm (your backend already does ROI work). Training on the palm ROI usually improves signal.
