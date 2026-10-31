"""Is It? - open-vocabulary visual verification on Gemma 4.
One image + one target -> {match, confidence, evidence}."""
import base64, hashlib, io, json, os, re, time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "gemini")
MODEL = os.getenv("MODEL", "gemma-4-26b-a4b-it")
FALLBACK = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")

PRESETS = {
    "free": {"label": "Anything (free text)", "target": "", "context": ""},
    "fish": {"label": "Fish freshness", "target": "fresh fish safe to buy",
             "context": "Check clear bulging eyes, red or pink gills, firm shiny skin. Cloudy eyes, grey gills, dull or slimy skin mean not fresh."},
    "leaf": {"label": "Crop leaf disease", "target": "a leaf showing blight, rust, or fungal disease symptoms",
             "context": "Look for lesions, spots with halos, rust-coloured pustules, powdery coating, or wilting. Healthy leaves are uniformly coloured."},
    "fraud": {"label": "Doctored receipt", "target": "a receipt or payment screenshot that shows signs of tampering",
              "context": "Look for mismatched fonts or alignment, inconsistent amounts, pasted regions, missing transaction IDs, odd timestamps."},
}
SCHEMA = {"type": "object", "required": ["match", "confidence", "evidence"],
          "properties": {"match": {"type": "boolean"}, "confidence": {"type": "number"},
                         "evidence": {"type": "string"}}}
RULES = ('You are a visual verifier. Decide whether the image matches the target. '
         'Reply with JSON only, no markdown: {"match": boolean, "confidence": number from 0 to 1, '
         '"evidence": string under 25 words naming what you see}. '
         'Use only visible evidence. If the image is unclear or the target is absent, lower the confidence.')

app = FastAPI(title="Is It?")
_cache: dict = {}
_client = None


def prep(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1024, 1024))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


def call_gemini(model: str, img: bytes, prompt: str) -> str:
    global _client
    from google import genai
    from google.genai import types
    _client = _client or genai.Client()
    r = _client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=img, mime_type="image/jpeg"), prompt])
    return r.text or ""


def call_ollama(img: bytes, prompt: str) -> str:
    r = httpx.post(f"{OLLAMA_URL}/api/chat", timeout=180, json={
        "model": OLLAMA_MODEL, "stream": False, "format": SCHEMA,
        "messages": [{"role": "user", "content": prompt,
                      "images": [base64.b64encode(img).decode()]}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("model returned no JSON")
    d = json.loads(m.group(0))
    return {"match": bool(d["match"]),
            "confidence": max(0.0, min(1.0, float(d["confidence"]))),
            "evidence": str(d.get("evidence", ""))[:300]}

"""
def run(img: bytes, prompt: str):
    if PROVIDER == "ollama":
        return parse(call_ollama(img, prompt)), f"ollama:{OLLAMA_MODEL}"
    try:
        return parse(call_gemini(MODEL, img, prompt)), MODEL
    except Exception as e:
        quota = any(s in str(e) for s in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND"))
        if quota and FALLBACK:
            return parse(call_gemini(FALLBACK, img, prompt)), FALLBACK
        raise
"""

def run(img: bytes, prompt: str):
    if PROVIDER == "ollama":
        return parse(call_ollama(img, prompt)), f"ollama:{OLLAMA_MODEL}"
    last = None
    for model in (MODEL, MODEL, FALLBACK):
        if not model:
            continue
        try:
            return parse(call_gemini(model, img, prompt)), model
        except Exception as e:
            last = e
            time.sleep(0.8)
    raise last

@app.get("/api/presets")
def presets():
    return [{"id": k, "label": v["label"], "target": v["target"]} for k, v in PRESETS.items()]


@app.get("/api/health")
def health():
    return {"ok": True, "provider": PROVIDER, "model": MODEL}


@app.post("/api/classify")
def classify(image: UploadFile = File(...), target: str = Form(""), preset: str = Form("free")):
    p = PRESETS.get(preset, PRESETS["free"])
    target = (target or p["target"]).strip()[:120]
    if not target:
        raise HTTPException(400, "Enter a target to check for.")
    raw = image.file.read()
    if len(raw) > 8_000_000:
        raise HTTPException(413, "Image is over 8 MB.")
    try:
        img = prep(raw)
    except Exception:
        raise HTTPException(400, "File is not a readable image.")
    prompt = f"{RULES}\nTarget: {target}\n" + (f"Guidance: {p['context']}\n" if p["context"] else "")
    key = hashlib.sha256(img + prompt.encode()).hexdigest()
    if key in _cache:
        return {**_cache[key], "cached": True}
    t0 = time.time()
    try:
        result, used = run(img, prompt)
    except Exception as e:
        raise HTTPException(502, f"Model call failed: {str(e)[:200]}")
    out = {**result, "target": target, "model": used, "ms": int((time.time() - t0) * 1000)}
    _cache[key] = out
    return {**out, "cached": False}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
