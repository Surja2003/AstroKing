# AstroKing Backend (FastAPI)

## Windows: MediaPipe-ready environment (Python 3.11)

MediaPipe wheels often lag behind bleeding-edge Python versions on Windows. Keep your system Python as-is and use an isolated Python 3.11 virtual environment for the backend.

### 1) Create the venv (Python launcher)

From the repo root:

```powershell
cd backend
py -3.11 -m venv palm_env
```

### 2) Install dependencies (without activation)

```powershell
.\palm_env\Scripts\python.exe -m pip install --upgrade pip
.\palm_env\Scripts\python.exe -m pip install -r requirements.txt
```

### 3) Run the API

```powershell
.\palm_env\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open docs at `http://127.0.0.1:8000/docs`.

## Palm scanning endpoints

- `POST /upload-palm` (JSON base64) — legacy / web-friendly
- `POST /scan-palm` (multipart) — preferred for mobile
- `POST /scan-palm/overlay` (multipart) — returns a JPEG overlay
- `POST /detect-hand-live` (multipart) — ultra-fast live detector for preview gating

## Personality (embedding-based) endpoints

This is a retrieval-based prototype that uses your trained CNN **embedding** model and a small curated archetype index.

- `POST /scan-palm/personality` (multipart) — returns archetype matches + traits (no DB write)
- `POST /scan-palm` and `POST /upload-palm` — now include a `personality` field
- `GET /personality/status` — quick debug/status probe (model path, index availability)

### Build the archetype index

From the repo root:

```powershell
cd ml
.\build_personality_index_windows.ps1
```

This writes `backend/palm_trait_index.npz` which the API loads automatically.

### Quick test (Windows)

With the API running:

```powershell
cd backend
.\test_personality_windows.ps1 -ImagePath "C:\path\to\palm.jpg"
```

### Config (optional)

- `PALM_EMBEDDING_MODEL`: path to a `.tflite` embedding model (defaults to `ml/models/hand_embedding_float16.tflite`)
- `PALM_TRAIT_INDEX`: path to the `.npz` index (defaults to `backend/palm_trait_index.npz`)

The backend also serves saved images at `/palm_images/...`.
