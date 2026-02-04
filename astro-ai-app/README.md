# AstroKing (Expo app)

This folder contains the AstroKing mobile app built with **Expo (React Native)** and **expo-router**.

For the overall repo overview (backend + ML + archetypes), see the root README: `../README.md`.

## Get started

1. Install dependencies

   ```bash
   npm install
   ```

2. Start the app

   ```bash
   npx expo start
   ```

## Native modules (AsyncStorage) — use a Dev Build

If you see `NativeModule: AsyncStorage is null`, you are running the app in **Expo Go**.
Expo Go can’t load arbitrary native modules; you need a **development build**.

One-time setup:

```bash
cd astro-ai-app
npx expo install @react-native-async-storage/async-storage
npx expo install expo-dev-client
```

Build + install the dev client:

```bash
# Android (needs a device or emulator + Android SDK configured)
npx expo run:android

# iOS (macOS only)
npx expo run:ios
```

Then start Metro for the dev client:

```bash
npx expo start --dev-client -c
```

## Backend (FastAPI)

This app expects a FastAPI backend.

- Default (recommended): `https://astroking.onrender.com`
- Local/LAN dev: set `EXPO_PUBLIC_API_URL` (see below)

Note: Render free tier sleeps after inactivity; the first request can take ~30–40s.

1. Start the backend (LAN-accessible for phones)

   ```bash
   cd ..\backend
   # Recommended on Windows for MediaPipe: use Python 3.11 venv
   py -3.11 -m venv palm_env
   .\palm_env\Scripts\python.exe -m pip install --upgrade pip
   .\palm_env\Scripts\python.exe -m pip install -r requirements.txt
   .\palm_env\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   If you prefer a helper script, see `backend/setup_windows.ps1`.

2. Verify routes

   - Laptop: `http://127.0.0.1:8000/docs`
   - Phone (same Wi‑Fi): `http://<YOUR_LAN_IP>:8000/docs`

### Palm scan (Computer Vision)

Two supported upload styles:

- JSON base64 (existing): `POST /upload-palm`
- Multipart (preferred for mobile): `POST /scan-palm` with `file` + optional `name`, `dob`

Optional overlay:

- `POST /scan-palm` with `overlay=true` saves a landmarks+ROI overlay image into `/palm_images` and returns `overlay_file`.
- `POST /scan-palm/overlay` returns a JPEG with landmarks drawn (no DB write).

3. If you’re testing on a physical device, set your LAN IP

   Copy [.env.example](.env.example) to `.env` and update (only needed for local/LAN dev):

   ```bash
   EXPO_PUBLIC_API_URL=http://<YOUR_LAN_IP>:8000
   ```

This project uses file-based routing via expo-router. Start with:
- `app/index.tsx`
- `app/(tabs)/_layout.tsx`
