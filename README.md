# AstroKing

AstroKing is a cross-platform **AI astrology + palm-scan companion** built with Expo/React Native and a FastAPI backend. It combines:
- An AI chat experience
- Palm/hand scanning (computer vision)
- Embedding-based “personality / archetype” matching
- Simple history & weekly/daily-style insight screens (app UI)

> Note: This project is intended for exploration/education and entertainment-style insights.

## Repo structure

- [`astro-ai-app/`](astro-ai-app/) — Expo (React Native) mobile app (iOS/Android/Web)
- [`backend/`](backend/) — FastAPI server for palm scan + personality endpoints
- [`ml/`](ml/) — training/evaluation/export tooling (TensorFlow/Keras + TFLite export)
- [`archetypes/`](archetypes/) — curated archetype definitions + metadata

## Tech stack

**Mobile app**
- Expo (React Native)
- expo-router (file-based routing)
- TypeScript + JavaScript

**Backend**
- Python + FastAPI (Uvicorn)
- MediaPipe + OpenCV for hand landmarks / CV pipeline
- NumPy, Pillow
- SQLAlchemy (data layer)

**ML workspace**
- TensorFlow / Keras
- scikit-learn + joblib
- TFLite export for lightweight inference

## Quick start

### 1) Run the backend (Windows-friendly)

The backend is happiest on **Python 3.11** on Windows (MediaPipe + TensorFlow compatibility).

```powershell
cd backend
py -3.11 -m venv palm_env
.\palm_env\Scripts\python.exe -m pip install --upgrade pip
.\palm_env\Scripts\python.exe -m pip install -r requirements.txt
.\palm_env\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs:
- http://127.0.0.1:8000/docs

More details: [`backend/README.md`](backend/README.md)

### 2) Run the Expo app

```bash
cd astro-ai-app
npm install
npx expo start
```

If you’re testing on a phone, set the backend URL:
- Copy [`astro-ai-app/.env.example`](astro-ai-app/.env.example) to `astro-ai-app/.env`
- Set `EXPO_PUBLIC_API_URL=http://<YOUR_LAN_IP>:8000`

### Dev build note (recommended)

Some features rely on native modules (e.g. image picker / device storage). For full functionality, use a **development build** instead of Expo Go.

```bash
cd astro-ai-app
npx expo install expo-dev-client
npx expo run:android
npx expo start --dev-client -c
```

## Palm scanning API (high level)

Backend endpoints include:
- `POST /scan-palm` (multipart) — preferred for mobile uploads
- `POST /upload-palm` (JSON base64) — web/legacy-friendly
- `POST /scan-palm/personality` — embedding-based archetype/trait matching

See [`backend/README.md`](backend/README.md) for the full list and notes.

## ML workspace

Training + export scripts live in [`ml/`](ml/). It supports training a small CNN and exporting an embedding model + TFLite.

See [`ml/README.md`](ml/README.md).
