from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import io
import os
from datetime import datetime, timedelta
from PIL import Image
from PIL import ImageFile
from typing import Any
from typing import List
from typing import Optional
import random
import requests
import sys
import threading
from textblob import TextBlob

from database import Base, SessionLocal, engine
import models as db_models

app = FastAPI()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AstroKing API",
        "docs": "/docs",
        "health": "/health",
        "hint": "Open /docs to view and test all endpoints.",
    }


@app.get("/personality/status")
def personality_status():
    """Debug/status endpoint for the embedding-based personality feature."""

    try:
        from palm_ml import get_embedding_dim, _get_model_path  # type: ignore
        from palm_personality import get_index_status

        model_path = _get_model_path()
        return {
            "model_path": model_path,
            "model_exists": os.path.exists(model_path),
            "embedding_dim": get_embedding_dim(),
            "index": get_index_status(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Create DB tables on startup (SQLite MVP).
Base.metadata.create_all(bind=engine)

# Serve saved palm images for the History screen.
_base_dir = os.path.dirname(__file__)
_palm_dir = os.path.join(_base_dir, "palm_images")
os.makedirs(_palm_dir, exist_ok=True)
app.mount("/palm_images", StaticFiles(directory=_palm_dir), name="palm_images")

# Dev CORS: allow the Expo web app (localhost) to call this API.
# Tighten this for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class User(BaseModel):
    name: str
    dob: str
    token: Optional[str] = None
    language: Optional[str] = None


def send_push_notification(token: str, message: str):
    url = "https://exp.host/--/api/v2/push/send"
    payload = {
        "to": token,
        "title": "Your Daily Insight",
        "body": message,
    }
    try:
        return requests.post(url, json=payload, timeout=10)
    except Exception:
        return None


class Chat(BaseModel):
    name: str
    dob: str
    message: str
    context: Optional[str] = None
    language: Optional[str] = None
    intent: Optional[str] = None
    recent_topics: Optional[List[str]] = None
    tone: Optional[str] = None


def _normalize_language(language: Optional[str]) -> str:
    l = (language or "").strip().lower()
    if l in {"hi", "hindi"}:
        return "hi"
    if l in {"bn", "bengali", "bangla"}:
        return "bn"
    return "en"


class PalmImage(BaseModel):
    image: str
    name: Optional[str] = None
    dob: Optional[str] = None


class SessionSummaryPayload(BaseModel):
    focus: str
    key_insight: str
    mood_trend: str
    reflection: str
    next_step: str


class SessionSummaryCreate(BaseModel):
    name: str
    dob: str
    summary: SessionSummaryPayload


def _get_or_create_user(db, *, name: str, dob: str, token: Optional[str] = None) -> db_models.User:
    user = db.query(db_models.User).filter_by(name=name, dob=dob).first()
    zodiac = get_zodiac(dob)

    if not user:
        user = db_models.User(name=name, dob=dob, zodiac=zodiac, push_token=token)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    updated = False
    if zodiac and user.zodiac != zodiac:
        user.zodiac = zodiac
        updated = True
    if token and user.push_token != token:
        user.push_token = token
        updated = True
    if updated:
        db.add(user)
        db.commit()
        db.refresh(user)

    return user


def _strip_data_url(image_str: str) -> str:
    s = (image_str or "").strip()
    if "," in s and s.lower().startswith("data:"):
        return s.split(",", 1)[1]
    return s


def _guess_ext(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return "bin"


def analyze_palm_image(image_bytes: bytes) -> dict[str, Any]:
    """Analyze an image of a palm/hand.

    Prefers MediaPipe Hands (stable hand landmark detection), and falls back to a
    lightweight OpenCV-only heuristic if MediaPipe isn't installed.
    """

    result: dict[str, Any] = {"features": {}, "traits": []}

    # 1) Prefer MediaPipe-based analysis.
    try:
        from palm_cv import analyze_palm_image as analyze_with_mp

        r = analyze_with_mp(image_bytes, return_overlay=False)
        return {
            "status": r.status,
            "features": r.features,
            "traits": r.traits,
        }
    except Exception:
        # Fall through to OpenCV-only heuristic.
        pass

    # 2) Fallback: OpenCV-only analysis (no landmarks).
    try:
        import numpy as np
        import cv2

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        edge_count = int(np.sum(edges > 0))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=40,
            maxLineGap=8,
        )
        line_count = int(0 if lines is None else len(lines))

        h, w = edges.shape[:2]
        pixels = float(max(1, h * w))
        density_score = float(edge_count / pixels)
        mean_gray = float(np.mean(blurred))

        result["features"] = {
            "edge_count": edge_count,
            "density_score": density_score,
            "mean_gray": mean_gray,
            "line_count": line_count,
            "image_size": {"width": int(w), "height": int(h)},
        }

        traits: list[str] = []
        if density_score > 0.15:
            traits.append("Highly active mind and energetic personality")
        else:
            traits.append("Calm thinker with steady decision-making")

        if mean_gray < 60:
            traits.append("Tip: brighter lighting improves scan accuracy")

        result["traits"] = traits
        result["status"] = "success"
        return result

    except Exception as e:
        msg = str(e)
        result["features"] = {"error": msg}
        result["traits"] = ["Image received, but analysis failed (try better lighting / focus)"]
        result["status"] = "error"
        return result


def analyze_palm_personality(image_bytes: bytes, *, top_k: int = 3) -> dict[str, Any]:
    """Optional ML-based personality matching.

    Uses the exported embedding model + a small curated nearest-neighbor index.
    If the model or index isn't available, returns status=unavailable.
    """

    try:
        from palm_ml import embed_image_bytes
        from palm_personality import match_personality

        emb = embed_image_bytes(image_bytes)
        if emb.status != "ok" or not emb.embedding:
            return {
                "status": emb.status,
                "error": emb.error,
                "matches": [],
                "traits": [],
            }

        return match_personality(emb.embedding, top_k=top_k)
    except Exception as e:
        return {"status": "error", "error": str(e), "matches": [], "traits": []}


@app.post("/scan-palm")
async def scan_palm(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    dob: Optional[str] = Form(None),
    overlay: bool = Form(False),
):
    """Multipart palm scan endpoint (preferred for mobile).

    Returns JSON analysis, and optionally saves an overlay image (landmarks + ROI) to
    the existing /palm_images static directory.
    """

    contents = await file.read()

    # Save the uploaded image (for History screen parity with /upload-palm).
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "palm_images")
    os.makedirs(out_dir, exist_ok=True)
    stamp = int(datetime.now().timestamp() * 1000)

    saved_rel_path: Optional[str] = None
    try:
        img = Image.open(io.BytesIO(contents))
        filename = os.path.join(out_dir, f"palm_{stamp}.png")
        img.save(filename)
        saved_rel_path = os.path.relpath(filename, base_dir).replace("\\", "/")
    except Exception:
        ext = _guess_ext(contents)
        filename = os.path.join(out_dir, f"palm_{stamp}.{ext}")
        with open(filename, "wb") as f:
            f.write(contents)
        saved_rel_path = os.path.relpath(filename, base_dir).replace("\\", "/")

    overlay_rel_path: Optional[str] = None
    if overlay:
        try:
            from palm_cv import analyze_palm_image as analyze_with_mp

            r = analyze_with_mp(contents, return_overlay=True)
            if r.overlay_jpeg_bytes:
                out_file = os.path.join(out_dir, f"palm_overlay_{stamp}.jpg")
                with open(out_file, "wb") as f:
                    f.write(r.overlay_jpeg_bytes)

                overlay_rel_path = os.path.relpath(out_file, base_dir).replace("\\", "/")
        except Exception:
            overlay_rel_path = None

    analysis = analyze_palm_image(contents)
    personality = analyze_palm_personality(contents, top_k=3)

    # If we have user identity, store the palm record.
    if name and dob and saved_rel_path:
        db = SessionLocal()
        try:
            user_db = _get_or_create_user(db, name=(name or "Friend").strip() or "Friend", dob=(dob or "").strip())
            record = db_models.PalmRecord(user_id=user_db.id, image_path=saved_rel_path)
            db.add(record)
            db.commit()
        finally:
            db.close()

    return {
        "status": analysis.get("status", "success"),
        "traits": analysis.get("traits", []),
        "analysis": analysis,
        "personality": personality,
        "file": saved_rel_path,
        "overlay_file": overlay_rel_path,
    }


@app.post("/scan-palm/personality")
async def scan_palm_personality(file: UploadFile = File(...), top_k: int = Form(3)):
    """Return ML personality matches only (no saving, no DB writes)."""
    contents = await file.read()
    return analyze_palm_personality(contents, top_k=max(1, min(int(top_k), 10)))


@app.post("/scan-palm/overlay")
async def scan_palm_overlay(file: UploadFile = File(...)):
    """Returns a JPEG with landmarks drawn (no DB write)."""
    try:
        from palm_cv import analyze_palm_image as analyze_with_mp

        contents = await file.read()
        r = analyze_with_mp(contents, return_overlay=True)
        if not r.overlay_jpeg_bytes:
            return Response(content=b"", media_type="image/jpeg", status_code=422)
        return Response(content=r.overlay_jpeg_bytes, media_type="image/jpeg")
    except Exception:
        return Response(content=b"", media_type="image/jpeg", status_code=500)


@app.post("/scan-palm/quality")
async def scan_palm_quality(file: UploadFile = File(...)):
    """Lightweight quality gate endpoint (no saving, no DB writes).

    Intended for preview-frame polling from the mobile app.
    """

    try:
        from palm_cv import assess_frame_quality

        contents = await file.read()
        q = assess_frame_quality(contents)
        return {
            "status": q.status,
            "message": q.message,
            "metrics": q.metrics,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": "Quality check failed",
            "metrics": {"error": str(e)},
        }


@app.post("/detect-hand-live")
async def detect_hand_live(file: UploadFile = File(...)):
    """Ultra-fast hand detector for camera preview gating.

    Returns `detected` plus an optional normalized bounding box.
    """

    try:
        from palm_cv import detect_hand_live as detect_live

        contents = await file.read()
        return detect_live(contents)
    except Exception as e:
        return {"detected": False, "reason": "error", "error": str(e)}


insights = [
    "Today is a good day to focus on communication.",
    "Take a small step toward your goal today.",
    "Stay calm, clarity will come.",
]

insights_by_language = {
    "en": insights,
    "hi": [
        "आज संचार पर ध्यान देने का अच्छा दिन है।",
        "आज अपने लक्ष्य की ओर एक छोटा कदम बढ़ाएँ।",
        "शांत रहें—स्पष्टता आएगी।",
    ],
    "bn": [
        "আজ যোগাযোগে মনোযোগ দেওয়ার ভালো দিন।",
        "আজ আপনার লক্ষ্যের দিকে ছোট্ট একটি পদক্ষেপ নিন।",
        "শান্ত থাকুন—স্পষ্টতা আসবে।",
    ],
}


def get_zodiac(dob: str) -> str:
    try:
        month, day = map(int, (dob or "").split("-")[1:])
    except Exception:
        return ""

    if (month == 3 and day >= 21) or (month == 4 and day <= 19):
        return "Aries"
    if (month == 4 and day >= 20) or (month == 5 and day <= 20):
        return "Taurus"
    if (month == 5 and day >= 21) or (month == 6 and day <= 20):
        return "Gemini"
    if (month == 6 and day >= 21) or (month == 7 and day <= 22):
        return "Cancer"
    if (month == 7 and day >= 23) or (month == 8 and day <= 22):
        return "Leo"
    if (month == 8 and day >= 23) or (month == 9 and day <= 22):
        return "Virgo"
    if (month == 9 and day >= 23) or (month == 10 and day <= 22):
        return "Libra"
    if (month == 10 and day >= 23) or (month == 11 and day <= 21):
        return "Scorpio"
    if (month == 11 and day >= 22) or (month == 12 and day <= 21):
        return "Sagittarius"
    if (month == 12 and day >= 22) or (month == 1 and day <= 19):
        return "Capricorn"
    if (month == 1 and day >= 20) or (month == 2 and day <= 18):
        return "Aquarius"
    return "Pisces"


reply_templates = {
    "general": [
        "{name}, trust yourself and take one clear step forward.",
        "{name}, slow down and choose the simplest next action.",
        "{name}, you're closer than you think-stay steady.",
        "{name}, clarity comes from action, not overthinking.",
    ],
    "money": [
        "{name}, choose one money move today: track, cut, or grow income.",
        "{name}, keep it simple: one budget rule, one action, repeat.",
        "{name}, stability builds from small consistent decisions - start with one.",
    ],
    "career": [
        "{name}, focus on one priority today - momentum will follow.",
        "{name}, a small improvement to your routine will compound fast.",
        "{name}, ask one direct question - communication is your advantage.",
    ],
    "love": [
        "{name}, be honest about what you need - softly, but clearly.",
        "{name}, give space for a calm conversation; don't rush the outcome.",
        "{name}, consistency matters more than grand gestures right now.",
    ],
    "mind": [
        "{name}, let's reduce the noise: name the worry, then choose one next step.",
        "{name}, clarity returns when you do one small grounding action today.",
        "{name}, keep it gentle: breathe, write one line, then act on one thing.",
    ],
    "health": [
        "{name}, prioritize rest and hydration - your energy will stabilize.",
        "{name}, keep it simple: move a little, eat a little better, repeat.",
        "{name}, listen to your body's signals - don't push past fatigue.",
    ],
}


reply_templates_by_language = {
    "en": reply_templates,
    "hi": {
        "general": [
            "{name}, खुद पर भरोसा रखें और आज एक साफ़ कदम आगे बढ़ाएँ।",
            "{name}, थोड़ा धीमे चलें और सबसे सरल अगला कदम चुनें।",
            "{name}, आप सोच से भी ज़्यादा करीब हैं—स्थिर रहें।",
            "{name}, स्पष्टता कार्रवाई से आती है, ज़्यादा सोचने से नहीं।",
        ],
        "money": [
            "{name}, आज पैसे के लिए एक कदम चुनें: ट्रैक करें, खर्च घटाएँ, या आय बढ़ाएँ।",
            "{name}, इसे सरल रखें: एक बजट नियम, एक कार्रवाई—और दोहराएँ।",
            "{name}, स्थिरता छोटे-छोटे फैसलों से बनती है—आज एक से शुरुआत करें।",
        ],
        "career": [
            "{name}, आज एक प्राथमिकता चुनें—गति अपने आप बनेगी।",
            "{name}, अपनी दिनचर्या में छोटा सुधार करें—यह जल्दी असर दिखाएगा।",
            "{name}, एक सीधा सवाल पूछें—आपकी ताकत संवाद है।",
        ],
        "love": [
            "{name}, जो चाहिए उसे नरमी से लेकिन साफ़ कहें।",
            "{name}, शांत बातचीत के लिए जगह दें; नतीजे को जल्दी मत करें।",
            "{name}, अभी बड़े इशारों से ज़्यादा निरंतरता मायने रखती है।",
        ],
        "mind": [
            "{name}, शोर कम करें: चिंता को नाम दें, फिर एक अगला कदम चुनें।",
            "{name}, आज एक छोटा ग्राउंडिंग कदम लें—स्पष्टता वापस आएगी।",
            "{name}, नरमी रखें: साँस लें, एक लाइन लिखें, फिर एक काम करें।",
        ],
        "health": [
            "{name}, आराम और पानी को प्राथमिकता दें—ऊर्जा स्थिर होगी।",
            "{name}, इसे सरल रखें: थोड़ा चलें, थोड़ा बेहतर खाएँ, दोहराएँ।",
            "{name}, शरीर के संकेत सुनें—थकान से आगे न बढ़ें।",
        ],
    },
    "bn": {
        "general": [
            "{name}, নিজের উপর বিশ্বাস রাখুন এবং আজ একটি পরিষ্কার পদক্ষেপ নিন।",
            "{name}, একটু ধীরে চলুন এবং সবচেয়ে সহজ পরের কাজটি বেছে নিন।",
            "{name}, আপনি ভাবছেন তার চেয়েও কাছাকাছি—স্থির থাকুন।",
            "{name}, অতিরিক্ত ভাবনা নয়—কাজের মাধ্যমেই স্পষ্টতা আসে।",
        ],
        "money": [
            "{name}, আজ টাকার জন্য একটিই পদক্ষেপ নিন: ট্র্যাক/খরচ কমানো/আয় বাড়ানো।",
            "{name}, সহজ রাখুন: একটি বাজেট নিয়ম, একটি কাজ—তারপর আবারও।",
            "{name}, স্থিরতা ছোট ধারাবাহিক সিদ্ধান্ত থেকে আসে—আজ একটিতে শুরু করুন।",
        ],
        "career": [
            "{name}, আজ একটি অগ্রাধিকার ঠিক করুন—গতিটা তৈরি হবে।",
            "{name}, রুটিনে ছোট্ট উন্নতি করুন—দ্রুত ফল পাবেন।",
            "{name}, একটি সরাসরি প্রশ্ন করুন—আপনার শক্তি হলো যোগাযোগ।",
        ],
        "love": [
            "{name}, যা দরকার তা কোমলভাবে কিন্তু স্পষ্ট করে বলুন।",
            "{name}, শান্ত কথোপকথনের জন্য সময় দিন; ফল তাড়াহুড়ো করবেন না।",
            "{name}, এখন বড় ইশারার চেয়ে ধারাবাহিকতা বেশি গুরুত্বপূর্ণ।",
        ],
        "mind": [
            "{name}, শব্দ কমাই: দুশ্চিন্তাটা নাম দিন, তারপর একটাই পরের পদক্ষেপ নিন।",
            "{name}, আজ একটি ছোট গ্রাউন্ডিং কাজ করুন—স্বচ্ছতা ফিরবে।",
            "{name}, কোমল থাকুন: শ্বাস নিন, এক লাইন লিখুন, তারপর এক কাজ করুন।",
        ],
        "health": [
            "{name}, বিশ্রাম আর পানি—এগুলোকে অগ্রাধিকার দিন; শক্তি স্থির হবে।",
            "{name}, সহজ রাখুন: একটু নড়াচড়া, একটু ভালো খাওয়া, আবারও।",
            "{name}, শরীরের সংকেত শুনুন—অতিরিক্ত চাপ দেবেন না।",
        ],
    },
}


followups_by_language = {
    "en": [
        "What's the one thing you can control about it?",
        "What outcome would feel like a win?",
        "Do you want a practical plan or a mindset check?",
    ],
    "hi": [
        "इसमें ऐसी एक चीज़ क्या है जिसे आप नियंत्रित कर सकते हैं?",
        "कौन सा परिणाम आपके लिए ‘जीत’ जैसा लगेगा?",
        "आपको व्यावहारिक योजना चाहिए या माइंडसेट चेक?",
    ],
    "bn": [
        "এটার কোন একটা বিষয় আপনি নিয়ন্ত্রণ করতে পারেন?",
        "কোন ফলাফলটা আপনার কাছে ‘জয়’ মনে হবে?",
        "আপনি বাস্তব পরিকল্পনা চান, নাকি মাইন্ডসেট চেক?",
    ],
}


context_prefix_by_language = {
    "en": "Since {context} is a focus for you right now, {reply}",
    "hi": "क्योंकि अभी आपका ध्यान {context} पर है, {reply}",
    "bn": "যেহেতু এখন আপনার ফোকাস {context} নিয়ে, {reply}",
}


context_aliases = {
    "your exams": {
        "hi": "आपकी परीक्षाएँ",
        "bn": "আপনার পরীক্ষা",
    },
    "your career": {
        "hi": "आपका करियर",
        "bn": "আপনার ক্যারিয়ার",
    },
    "your relationships": {
        "hi": "आपके रिश्ते",
        "bn": "আপনার সম্পর্ক",
    },
    "your stress levels": {
        "hi": "आपका तनाव",
        "bn": "আপনার স্ট্রেস",
    },
    "your health and energy": {
        "hi": "आपका स्वास्थ्य और ऊर्जा",
        "bn": "আপনার স্বাস্থ্য ও শক্তি",
    },
    "your finances": {
        "hi": "आपकी वित्तीय स्थिति",
        "bn": "আপনার আর্থিক অবস্থা",
    },
    "your mental clarity": {
        "hi": "आपकी मानसिक स्पष्टता",
        "bn": "আপনার মানসিক স্বচ্ছতা",
    },
}


def _normalize_intent(intent: Optional[str]) -> str:
    i = (intent or "").strip().lower()
    if i in {"money", "love", "mind", "career"}:
        return i
    return "general"


def _normalize_tone(tone: Optional[str]) -> str:
    t = (tone or "").strip().lower()
    if t in {"supportive", "motivational"}:
        return t
    return "balanced"


def _context_from_intent(intent: str) -> str:
    if intent == "money":
        return "your finances"
    if intent == "love":
        return "your relationships"
    if intent == "mind":
        return "your mental clarity"
    if intent == "career":
        return "your career"
    return ""


def _localize_context(context: str, language: str) -> str:
    c = (context or "").strip()
    if not c:
        return ""
    if language == "en":
        return c
    key = c.lower()
    mapped = context_aliases.get(key, {}).get(language)
    return mapped or c


def _pick_category(message: str, intent: Optional[str] = None) -> str:
    forced = _normalize_intent(intent)
    if forced != "general":
        return forced

    m = (message or "").lower()
    if any(k in m for k in ["money", "finance", "finances", "debt", "loan", "emi", "salary", "budget", "invest"]):
        return "money"
    if any(k in m for k in ["job", "work", "career", "boss", "interview", "promotion", "resume", "portfolio", "business"]):
        return "career"
    if any(k in m for k in ["love", "relationship", "crush", "partner", "breakup", "marriage", "date"]):
        return "love"
    if any(k in m for k in ["stress", "sleep", "anxiety", "worry", "overthink", "panic", "mental"]):
        return "mind"
    if any(k in m for k in ["health", "diet", "workout", "gym", "tired", "energy", "sleep"]):
        return "health"
    return "general"


def generate_reply(name: str, message: str, language: str = "en", intent: Optional[str] = None, tone: Optional[str] = None) -> str:
    category = _pick_category(message, intent=intent)
    templates = reply_templates_by_language.get(language) or reply_templates
    bucket = templates.get(category) or templates.get("general") or reply_templates["general"]
    base = random.choice(bucket).format(name=name)

    tone_key = _normalize_tone(tone)
    tone_prefixes = {
        "en": {
            "supportive": "I'll keep this gentle and clear. ",
            "motivational": "Let's keep this focused and actionable. ",
            "balanced": "",
        },
        "hi": {
            "supportive": "मैं इसे नरमी और स्पष्टता के साथ कहूँगा/कहूँगी। ",
            "motivational": "चलो इसे फोकस्ड और एक्शन-ओरिएंटेड रखें। ",
            "balanced": "",
        },
        "bn": {
            "supportive": "আমি এটা কোমলভাবে আর পরিষ্কারভাবে বলি। ",
            "motivational": "চলুন এটাকে ফোকাসড এবং অ্যাকশনেবল রাখি। ",
            "balanced": "",
        },
    }
    prefix = (tone_prefixes.get(language) or tone_prefixes["en"]).get(tone_key) or ""
    base = f"{prefix}{base}" if prefix else base

    # Add a little variety if the user asked a question.
    if (message or "").strip().endswith("?"):
        followups = followups_by_language.get(language) or followups_by_language["en"]
        return f"{base} {random.choice(followups)}"

    return base


def detect_emotion(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["stress", "stressed", "anxious", "anxiety", "overwhelmed", "panic", "panicking", "stressing", "freaking out"]):
        return "anxiety"
    if any(k in t for k in ["nothing is working", "nothing works", "hopeless", "depressed", "sad", "down", "heartbroken", "miserable", "lonely"]):
        return "sad"
    if any(k in t for k in ["excited", "happy", "great", "amazing", "awesome", "thrilled"]):
        return "happy"

    try:
        analysis = TextBlob(text or "")
        polarity = float(analysis.sentiment.polarity)
    except Exception:
        return "neutral"

    if polarity > 0.3:
        return "happy"
    if polarity < -0.3:
        return "sad"
    return "neutral"


@app.post("/daily-insight")
def daily_insight(user: User):
    name = (user.name or "Friend").strip() or "Friend"
    dob = (user.dob or "").strip()
    token = (user.token or "").strip() or None
    language = _normalize_language(getattr(user, "language", None))

    db = SessionLocal()
    try:
        user_db = _get_or_create_user(db, name=name, dob=dob, token=token)
        zodiac = user_db.zodiac or get_zodiac(dob)
        push_token = user_db.push_token
    finally:
        db.close()

    zodiac_insights_en = {
        "Aries": "Take bold action today, but pause before reacting.",
        "Taurus": "Stability comes from patience today.",
        "Gemini": "Communication opens new doors today.",
        "Cancer": "Focus on emotional balance and home energy.",
        "Leo": "Your confidence attracts attention today.",
        "Virgo": "Small improvements compound - organize one thing today.",
        "Libra": "Choose harmony, but don't avoid the truth.",
        "Scorpio": "Go deep - one honest conversation shifts everything.",
        "Sagittarius": "Say yes to learning or exploring something new.",
        "Capricorn": "Discipline beats motivation - keep it simple and steady.",
        "Aquarius": "A fresh idea helps - share it with the right person.",
        "Pisces": "Trust your intuition today.",
    }
    zodiac_insights_hi = {
        "Aries": "आज साहसिक कदम उठाएँ, लेकिन प्रतिक्रिया देने से पहले रुकें।",
        "Taurus": "आज स्थिरता धैर्य से आएगी।",
        "Gemini": "आज संवाद नए दरवाज़े खोल सकता है।",
        "Cancer": "भावनात्मक संतुलन और घर की ऊर्जा पर ध्यान दें।",
        "Leo": "आज आपका आत्मविश्वास ध्यान खींचेगा।",
        "Virgo": "छोटे सुधार बड़ा असर करते हैं—आज एक चीज़ व्यवस्थित करें।",
        "Libra": "संतुलन चुनें, लेकिन सच से बचें नहीं।",
        "Scorpio": "गहराई में जाएँ—एक ईमानदार बातचीत सब बदल सकती है।",
        "Sagittarius": "सीखने या कुछ नया खोजने के लिए ‘हाँ’ कहें।",
        "Capricorn": "अनुशासन प्रेरणा से बेहतर है—सरल और स्थिर रहें।",
        "Aquarius": "एक नया विचार मदद करेगा—सही व्यक्ति से साझा करें।",
        "Pisces": "आज अपनी अंतर्ज्ञान पर भरोसा रखें।",
    }
    zodiac_insights_bn = {
        "Aries": "আজ সাহসী পদক্ষেপ নিন, তবে প্রতিক্রিয়া দেওয়ার আগে একটু থামুন।",
        "Taurus": "আজ ধৈর্যই স্থিরতা এনে দেবে।",
        "Gemini": "আজ যোগাযোগ নতুন সুযোগ খুলে দিতে পারে।",
        "Cancer": "আবেগের ভারসাম্য এবং ঘরের শক্তিতে মন দিন।",
        "Leo": "আজ আপনার আত্মবিশ্বাস নজর কাড়বে।",
        "Virgo": "ছোট উন্নতিই বড় ফল দেয়—আজ একটি কাজ গুছিয়ে নিন।",
        "Libra": "সমতা বেছে নিন, কিন্তু সত্য এড়াবেন না।",
        "Scorpio": "গভীরে যান—একটা সৎ কথোপকথন অনেক বদলে দিতে পারে।",
        "Sagittarius": "শেখা বা নতুন কিছু অন্বেষণ করতে ‘হ্যাঁ’ বলুন।",
        "Capricorn": "প্রেরণার চেয়ে শৃঙ্খলাই শক্তিশালী—সহজ ও স্থির থাকুন।",
        "Aquarius": "একটা নতুন আইডিয়া সাহায্য করবে—সঠিক মানুষের সাথে শেয়ার করুন।",
        "Pisces": "আজ আপনার অন্তর্দৃষ্টিতে ভরসা রাখুন।",
    }

    zodiac_insights_by_language = {
        "en": zodiac_insights_en,
        "hi": zodiac_insights_hi,
        "bn": zodiac_insights_bn,
    }

    zodiac_insights = zodiac_insights_by_language.get(language) or zodiac_insights_en
    fallback_insights = insights_by_language.get(language) or insights
    insight = zodiac_insights.get(zodiac) or random.choice(fallback_insights)

    if push_token:
        title = "Your Daily Insight"
        if language == "hi":
            title = "आपका आज का संदेश"
        elif language == "bn":
            title = "আজকের বার্তা"

        url = "https://exp.host/--/api/v2/push/send"
        payload = {
            "to": push_token,
            "title": title,
            "body": insight,
        }
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception:
            pass

    return {"zodiac": zodiac, "insight": insight}


@app.post("/chat")
def chat(chat: Chat):
    name = (chat.name or "Friend").strip() or "Friend"
    message = (chat.message or "").strip()
    context = (chat.context or "").strip()
    language = _normalize_language(getattr(chat, "language", None))
    intent = _normalize_intent(getattr(chat, "intent", None))
    tone = _normalize_tone(getattr(chat, "tone", None))
    recent_topics = getattr(chat, "recent_topics", None) or []
    if context:
        context = " ".join(context.split())
        context = context.replace("\n", " ").replace("\r", " ")
        context = context[:80]

    derived_context = ""
    if not context:
        if intent and intent != "general":
            derived_context = _context_from_intent(intent)
        elif isinstance(recent_topics, list):
            for item in recent_topics:
                i = _normalize_intent(str(item))
                if i != "general":
                    derived_context = _context_from_intent(i)
                    break

    final_context = context or derived_context
    emotion = detect_emotion(message)

    if emotion == "happy":
        if language == "hi":
            reply = f"यह सुनकर बहुत अच्छा लगा, {name}! इसी सकारात्मक ऊर्जा को बनाए रखें।"
        elif language == "bn":
            reply = f"দারুণ খবর, {name}! এই ইতিবাচক শক্তিটা ধরে রাখুন।"
        else:
            reply = f"That's wonderful to hear, {name}! Keep that positive energy flowing."
    elif emotion == "anxiety":
        if language == "hi":
            reply = f"मैं आपके साथ हूँ, {name}. धीरे-धीरे साँस लें—हम इसे सरल रखेंगे: एक छोटा कदम, फिर अगला।"
        elif language == "bn":
            reply = f"আমি আছি, {name}. ধীরে শ্বাস নিন—সহজ রাখি: এক ছোট পদক্ষেপ, তারপর আরেকটা।"
        else:
            reply = (
                f"I've got you, {name}. Take a slow breath - "
                "we'll keep it simple: one small step, then the next."
            )
    elif emotion == "sad":
        if language == "hi":
            reply = f"मैं यहाँ हूँ, {name}. मुश्किल समय टिकता नहीं—आज बस एक छोटा कदम आगे बढ़ाएँ।"
        elif language == "bn":
            reply = f"আমি আপনার পাশে আছি, {name}. কঠিন সময় স্থায়ী নয়—আজ শুধু এক ছোট পদক্ষেপ নিন।"
        else:
            reply = f"I'm here for you, {name}. Tough moments don't last - take one small step forward."
    else:
        reply = generate_reply(name=name, message=message, language=language, intent=intent, tone=tone)

    if final_context:
        localized_context = _localize_context(final_context, language)
        lowered = reply.lower()
        if localized_context.lower() not in lowered:
            prefix = context_prefix_by_language.get(language) or context_prefix_by_language["en"]
            reply = prefix.format(context=localized_context, reply=reply)

    db = SessionLocal()
    try:
        user_db = _get_or_create_user(db, name=name, dob=(chat.dob or "").strip())
        chat_entry = db_models.ChatHistory(user_id=user_db.id, message=message, reply=reply, emotion=emotion)
        db.add(chat_entry)
        db.commit()
    finally:
        db.close()

    return {"reply": reply, "emotion": emotion}


@app.post("/upload-palm")
def upload_palm(data: PalmImage):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    b64 = _strip_data_url(data.image)
    img_bytes = base64.b64decode(b64)

    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "palm_images")
    os.makedirs(out_dir, exist_ok=True)

    stamp = int(datetime.now().timestamp() * 1000)

    analysis = analyze_palm_image(img_bytes)
    personality = analyze_palm_personality(img_bytes, top_k=3)

    # Prefer decoding + re-encoding via Pillow (gives consistent PNG output),
    # but fall back to saving raw bytes if the image stream is imperfect.
    try:
        img = Image.open(io.BytesIO(img_bytes))
        filename = os.path.join(out_dir, f"palm_{stamp}.png")
        img.save(filename)
        out_file = filename
    except Exception:
        ext = _guess_ext(img_bytes)
        filename = os.path.join(out_dir, f"palm_{stamp}.{ext}")
        with open(filename, "wb") as f:
            f.write(img_bytes)
        out_file = filename

    rel_path = os.path.relpath(out_file, base_dir)
    # Normalize for URLs + cross-platform consistency.
    rel_path = rel_path.replace("\\", "/")

    # If we have user identity, store the palm record.
    if data.name and data.dob:
        db = SessionLocal()
        try:
            user_db = _get_or_create_user(
                db,
                name=(data.name or "Friend").strip() or "Friend",
                dob=(data.dob or "").strip(),
            )
            record = db_models.PalmRecord(user_id=user_db.id, image_path=rel_path)
            db.add(record)
            db.commit()
        finally:
            db.close()

    return {
        "status": "saved",
        "file": rel_path,
        "traits": analysis.get("traits", []),
        "analysis": analysis,
        "personality": personality,
    }


@app.get("/history/chat")
def get_chat_history(name: str, dob: str, limit: int = 50):
    safe_limit = max(1, min(int(limit), 200))
    db = SessionLocal()
    try:
        user_db = db.query(db_models.User).filter_by(name=name, dob=dob).first()
        if not user_db:
            return {"items": []}

        rows = (
            db.query(db_models.ChatHistory)
            .filter_by(user_id=user_db.id)
            .order_by(db_models.ChatHistory.id.desc())
            .limit(safe_limit)
            .all()
        )

        items = [
            {
                "id": r.id,
                "message": r.message,
                "reply": r.reply,
                "emotion": r.emotion,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
        return {"items": items}
    finally:
        db.close()


@app.get("/history/palms")
def get_palm_history(request: Request, name: str, dob: str, limit: int = 50):
    safe_limit = max(1, min(int(limit), 200))
    db = SessionLocal()
    try:
        user_db = db.query(db_models.User).filter_by(name=name, dob=dob).first()
        if not user_db:
            return {"items": []}

        rows = (
            db.query(db_models.PalmRecord)
            .filter_by(user_id=user_db.id)
            .order_by(db_models.PalmRecord.id.desc())
            .limit(safe_limit)
            .all()
        )

        base_url = str(request.base_url).rstrip("/")
        items = []
        for r in rows:
            image_path = (r.image_path or "").replace("\\", "/")
            image_url = f"{base_url}/{image_path.lstrip('/')}" if image_path else None
            items.append(
                {
                    "id": r.id,
                    "image_path": image_path,
                    "image_url": image_url,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return {"items": items}
    finally:
        db.close()


@app.post("/session-summary")
def create_session_summary(payload: SessionSummaryCreate):
    name = (payload.name or "Friend").strip() or "Friend"
    dob = (payload.dob or "").strip()

    s = payload.summary
    focus = (s.focus or "").strip() or "General"
    key_insight = (s.key_insight or "").strip() or "—"
    mood_trend = (s.mood_trend or "").strip() or "—"
    reflection = (s.reflection or "").strip() or "—"
    next_step = (s.next_step or "").strip() or "—"

    db = SessionLocal()
    try:
        user_db = _get_or_create_user(db, name=name, dob=dob)
        row = db_models.SessionSummary(
            user_id=user_db.id,
            focus=focus,
            key_insight=key_insight,
            mood_trend=mood_trend,
            reflection=reflection,
            next_step=next_step,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return {
            "id": row.id,
            "focus": row.focus,
            "key_insight": row.key_insight,
            "mood_trend": row.mood_trend,
            "reflection": row.reflection,
            "next_step": row.next_step,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
    finally:
        db.close()


@app.get("/history/summaries")
def get_session_summaries(name: str, dob: str, limit: int = 50):
    safe_limit = max(1, min(int(limit), 200))
    db = SessionLocal()
    try:
        user_db = db.query(db_models.User).filter_by(name=name, dob=dob).first()
        if not user_db:
            return {"items": []}

        rows = (
            db.query(db_models.SessionSummary)
            .filter_by(user_id=user_db.id)
            .order_by(db_models.SessionSummary.id.desc())
            .limit(safe_limit)
            .all()
        )

        items = [
            {
                "id": r.id,
                "focus": r.focus,
                "key_insight": r.key_insight,
                "mood_trend": r.mood_trend,
                "reflection": r.reflection,
                "next_step": r.next_step,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
        return {"items": items}
    finally:
        db.close()


@app.get("/debug-env")
def debug_env():
    try:
        import cv2

        cv2_status: Any = {"ok": True, "version": getattr(cv2, "__version__", None)}
    except Exception as e:
        cv2_status = {"ok": False, "error": str(e)}

    return {
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "cv2": cv2_status,
    }


def _classify_topic(message: str) -> str:
    t = (message or "").lower()
    if any(k in t for k in ["money", "finance", "finances", "debt", "save", "savings", "budget", "invest", "salary"]):
        return "money"
    if any(k in t for k in ["love", "relationship", "dating", "crush", "partner", "breakup", "marriage"]):
        return "love"
    if any(k in t for k in ["career", "job", "work", "boss", "promotion", "interview", "resume", "business"]):
        return "career"
    if any(k in t for k in ["anxiety", "stress", "overwhelmed", "panic", "focus", "mind", "confidence", "motivation"]):
        return "mind"
    if any(k in t for k in ["health", "sleep", "workout", "gym", "diet", "energy", "tired", "burnout"]):
        return "health"
    return "general"


@app.get("/weekly-report")
def weekly_report(name: str, dob: str, days: int = 7, language: Optional[str] = None):
    lang = _normalize_language(language)
    safe_days = max(3, min(int(days), 30))
    since = datetime.utcnow() - timedelta(days=safe_days)
    today_key = datetime.utcnow().date().isoformat()

    db = SessionLocal()
    try:
        user_db = db.query(db_models.User).filter_by(name=name, dob=dob).first()
        if not user_db:
            empty_msg = "Set your profile and start a few chats to unlock your weekly report."
            if lang == "hi":
                empty_msg = "अपनी प्रोफ़ाइल सेट करें और कुछ चैट शुरू करें—तभी आपका साप्ताहिक रिपोर्ट खुलेगा।"
            elif lang == "bn":
                empty_msg = "প্রোফাইল সেট করুন এবং কয়েকটি চ্যাট শুরু করুন—তাহলেই সাপ্তাহিক রিপোর্ট পাবেন।"
            return {
                "days": safe_days,
                "range": {"from": since.date().isoformat(), "to": today_key},
                "total_messages": 0,
                "active_days": 0,
                "streak_days": 0,
                "topics": {"general": 0},
                "moods": {"neutral": 0},
                "trend": [],
                "most_asked": "general",
                "key_insight": empty_msg,
            }

        rows = (
            db.query(db_models.ChatHistory)
            .filter(db_models.ChatHistory.user_id == user_db.id)
            .filter(db_models.ChatHistory.created_at >= since)
            .order_by(db_models.ChatHistory.created_at.asc())
            .all()
        )

        active_date_set: set[str] = set()

        topics: dict[str, int] = {}
        moods: dict[str, int] = {}
        by_day: dict[str, dict[str, int]] = {}

        for r in rows:
            topic = _classify_topic(r.message)
            emotion = (r.emotion or "neutral").lower() or "neutral"

            topics[topic] = topics.get(topic, 0) + 1
            moods[emotion] = moods.get(emotion, 0) + 1

            day_key = (r.created_at or datetime.utcnow()).date().isoformat()
            active_date_set.add(day_key)
            if day_key not in by_day:
                by_day[day_key] = {"happy": 0, "neutral": 0, "sad": 0, "anxiety": 0}
            if emotion not in by_day[day_key]:
                by_day[day_key][emotion] = 0
            by_day[day_key][emotion] += 1

        # Determine the dominant mood per day.
        trend = []
        for day_key in sorted(by_day.keys()):
            counts = by_day[day_key]
            dominant = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "neutral"
            trend.append({"date": day_key, "mood": dominant, "counts": counts})

        # Current streak = consecutive active days ending today.
        streak = 0
        for i in range(safe_days):
            d = (datetime.utcnow().date() - timedelta(days=i)).isoformat()
            if d in active_date_set:
                streak += 1
            else:
                break

        most_asked = "general"
        if topics:
            most_asked = max(topics.items(), key=lambda kv: kv[1])[0]

        zodiac = user_db.zodiac or get_zodiac(dob)
        # A simple "key insight" that blends topic + mood.
        if lang == "hi":
            if most_asked == "career":
                base = "आपका सप्ताह संरचना से प्रगति दिखा रहा है: एक स्पष्ट प्राथमिकता, बहुत कुछ साथ करने से बेहतर है।"
            elif most_asked == "money":
                base = "आपके सप्ताह को स्पष्टता चाहिए: नियंत्रण पाने के लिए रोज़ एक संख्या ट्रैक करें (खर्च या बचत)।"
            elif most_asked == "love":
                base = "आपका सप्ताह गर्मजोशी के साथ ईमानदारी का है: जो कहना है, नरमी से और साफ़ कहें।"
            elif most_asked == "mind":
                base = "आपका सप्ताह नियंत्रित गति का है: अगला कदम सरल करें और ध्यान की रक्षा करें।"
            elif most_asked == "health":
                base = "आपका सप्ताह ऊर्जा प्रबंधन का है: नींद और नियमित दिनचर्या जल्दी असर दिखाएगी।"
            else:
                base = "आपका सप्ताह निरंतरता का है: रोज़ के छोटे कदम असली बदलाव बनाते हैं।"
        elif lang == "bn":
            if most_asked == "career":
                base = "এই সপ্তাহে কাঠামোই অগ্রগতির চাবিকাঠি: একটাই স্পষ্ট অগ্রাধিকার, অনেক কিছু সামলানোর চেয়ে ভালো।"
            elif most_asked == "money":
                base = "এই সপ্তাহে স্পষ্টতা দরকার: নিয়ন্ত্রণ ফেরাতে প্রতিদিন একটাই সংখ্যা ট্র্যাক করুন (খরচ বা সঞ্চয়)।"
            elif most_asked == "love":
                base = "এই সপ্তাহটা উষ্ণতার সাথে সততার: যা বলতে চান, কোমলভাবে কিন্তু স্পষ্ট করে বলুন।"
            elif most_asked == "mind":
                base = "এই সপ্তাহটা নিয়ন্ত্রিত গতি: পরের পদক্ষেপ সহজ করুন এবং মনোযোগ রক্ষা করুন।"
            elif most_asked == "health":
                base = "এই সপ্তাহটা শক্তি ব্যবস্থাপনা: ঘুম আর নিয়মিত রুটিন দ্রুত ফল দেবে।"
            else:
                base = "এই সপ্তাহটা ধারাবাহিকতার: প্রতিদিনের ছোট কাজই সত্যিকারের পরিবর্তন আনে।"
        else:
            if most_asked == "career":
                base = "Your week is pointing to progress through structure: one clear priority beats juggling."
            elif most_asked == "money":
                base = "Your week is asking for clarity: track one number daily (spend or save) to regain control."
            elif most_asked == "love":
                base = "Your week is about honesty with warmth: say what you mean, gently and directly."
            elif most_asked == "mind":
                base = "Your week is about controlled momentum: simplify your next step and protect your attention."
            elif most_asked == "health":
                base = "Your week is about energy management: sleep and a consistent routine will compound quickly."
            else:
                base = "Your week is about consistency: small actions daily create real change."

        mood_note = ""
        if moods.get("anxiety", 0) >= max(1, moods.get("happy", 0)):
            if lang == "hi":
                mood_note = " अगर तनाव बढ़ा हो, तो फैसले छोटे और दोहराए जा सकने वाले रखें।"
            elif lang == "bn":
                mood_note = " যদি স্ট্রেস বেড়ে থাকে, সিদ্ধান্তগুলো ছোট ও পুনরাবৃত্তিযোগ্য রাখুন।"
            else:
                mood_note = " If stress spiked, keep decisions smaller and repeatable."
        elif moods.get("happy", 0) > moods.get("sad", 0) + moods.get("anxiety", 0):
            if lang == "hi":
                mood_note = " आपकी गति मजबूत है—एक सार्थक चुनौती पर ध्यान दें।"
            elif lang == "bn":
                mood_note = " আপনার গতি ভালো—একটা অর্থপূর্ণ চ্যালেঞ্জ নিন।"
            else:
                mood_note = " Your momentum is strong—lean into one meaningful challenge."

        key_insight = f"{base}{mood_note}"
        if zodiac:
            key_insight = f"{zodiac}: {key_insight}"

        return {
            "days": safe_days,
            "range": {"from": since.date().isoformat(), "to": today_key},
            "total_messages": len(rows),
            "active_days": len(active_date_set),
            "streak_days": streak,
            "topics": topics or {"general": 0},
            "moods": moods or {"neutral": 0},
            "trend": trend,
            "most_asked": most_asked,
            "key_insight": key_insight,
        }
    finally:
        db.close()


@app.post("/__shutdown")
def __shutdown():
    """Dev-only: stop the server process (useful if a stray process holds the port)."""

    def _exit():
        os._exit(0)

    threading.Timer(0.2, _exit).start()
    return {"status": "shutting_down"}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
