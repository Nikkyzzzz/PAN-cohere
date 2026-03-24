import streamlit as st
import fitz  # PyMuPDF — no poppler needed
import cohere
import pandas as pd
import json
import re
import io
import os
import importlib.util
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv

# Skip slow Paddle model-host connectivity check during startup.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import easyocr
except Exception:
    easyocr = None

PaddleOCR = None

def paddle_is_available() -> bool:
    if PaddleOCR is not None:
        return True
    return importlib.util.find_spec("paddleocr") is not None

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

def get_secret(key: str, default: str = ""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

COHERE_API_KEY = get_secret("COHERE_API_KEY")
COHERE_MODEL_ENV = get_secret("COHERE_MODEL")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PAN Card Data Extractor",
    page_icon="🪪",
    layout="wide",
)

# ── HARDCODED OPTIMIZED VALUES ────────────────────────────────────────────────
# These are optimized for speed while maintaining decent accuracy
OCR_DPI = 150  # Lower DPI = faster processing
OCR_MAX_SIDE = 800  # Smaller side limit gives a strong speed boost
COHERE_MODEL = "command-r-08-2024"  # Faster model
OCR_ENGINE = "EasyOCR"  # Use fastest OCR engine
USE_PDF_TEXT_FIRST = True  # Use embedded text when available
ALLOW_PADDLE_FALLBACK = False  # Disable slower fallback

# ── Styling (minimal) ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .hero {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: white;
    }
    .hero h1 { font-size: 1.8rem; font-weight: 800; margin-bottom: 0; }
    .hero p { font-size: 0.9rem; opacity: 0.85; margin: 0; }
    .card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    .field-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #7986cb;
    }
    .field-value {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a237e;
    }
    .pan-badge {
        display: inline-block;
        background: #e8eaf6;
        color: #283593;
        font-weight: 800;
        font-size: 1rem;
        border-radius: 8px;
        padding: 2px 12px;
        font-family: monospace;
    }
    .metric-box {
        background: #e8eaf6;
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
    }
    .metric-box .num { font-size: 1.5rem; font-weight: 800; color: #1a237e; }
    .metric-box .lbl { font-size: 0.75rem; color: #5c6bc0; }
</style>
""", unsafe_allow_html=True)

# ── Hero (minimal) ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🪪 PAN Card Data Extractor</h1>
    <p>Upload PDF · Auto-extract PAN details · Export CSV</p>
</div>
""", unsafe_allow_html=True)

# Show only essential status
if not COHERE_API_KEY:
    st.error("❌ COHERE_API_KEY missing — please add to .env")
    st.stop()

if easyocr is None and not paddle_is_available():
    st.error("No OCR engine available. Install EasyOCR.")
    st.stop()

# ── PDF → PIL images via PyMuPDF ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def pdf_to_images_and_text(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = OCR_DPI / 72
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    for page in doc:
        page_text = page.get_text("text") or ""
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append({"image": img, "pdf_text": page_text})
    doc.close()
    return pages

# ── OCR (optimized) ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_easyocr_reader():
    if easyocr is None:
        return None
    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False,
        detector=True,
        recognizer=True,
    )

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Minimal preprocessing for speed."""
    w, h = img.size
    longest = max(w, h)
    if longest > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / float(longest)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    # Skip heavy processing - just convert to grayscale
    return img.convert("L")

def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _ocr_image_impl(img: Image.Image) -> str:
    """Optimized OCR - uses only EasyOCR for speed."""
    pre = preprocess_image_for_ocr(img)
    gray = np.array(pre)
    
    reader = get_easyocr_reader()
    if reader is None:
        return ""
    
    result = reader.readtext(gray, detail=0, batch_size=4)  # batch OCR for throughput
    return "\n".join(result)

@st.cache_data(show_spinner=False)
def ocr_image_cached(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _ocr_image_impl(img)

def is_pdf_text_usable(text: str) -> bool:
    t = (text or "").upper()
    if len(t.strip()) < 40:
        return False
    markers = ["INCOME", "PERMANENT", "ACCOUNT", "NUMBER", "GOVT", "NAME"]
    return any(m in t for m in markers)

# ── Cohere extraction (optimized prompt) ──────────────────────────────────────
SYSTEM_PROMPT = """Extract PAN card details from OCR text. Return JSON array with fields: pan_number, name, fathers_name, date_of_birth. Use null if missing. No markdown."""

PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
PAN_CANDIDATE_RE = re.compile(r"[A-Z0-9]{8,16}", re.IGNORECASE)
DOB_DMY_RE = re.compile(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b")
DOB_YMD_RE = re.compile(r"\b(\d{4}[/-]\d{2}[/-]\d{2})\b")

def normalize_pan_candidate(token: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]", "", str(token or "")).upper()
    if len(clean) != 10:
        return ""
    
    to_letter = {"0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
                 "5": "S", "6": "G", "7": "T", "8": "B", "9": "G"}
    to_digit = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "1", "J": "1",
                "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8", "R": "8"}
    
    chars = list(clean)
    for i in range(5):
        chars[i] = to_letter.get(chars[i], chars[i])
    for i in range(5, 9):
        chars[i] = to_digit.get(chars[i], chars[i])
    chars[9] = to_letter.get(chars[9], chars[9])
    norm = "".join(chars)
    return norm if PAN_RE.match(norm) else ""

def extract_pan_candidates_from_text(text: str) -> list[str]:
    candidates = []
    seen = set()
    tokens = PAN_CANDIDATE_RE.findall((text or "").upper())
    for token in tokens:
        clean = re.sub(r"[^A-Z0-9]", "", token)
        if len(clean) < 10:
            continue
        windows = [clean] if len(clean) == 10 else [clean[i:i+10] for i in range(0, len(clean)-9)]
        for win in windows:
            pan = normalize_pan_candidate(win)
            if pan and pan not in seen:
                seen.add(pan)
                candidates.append(pan)
    return candidates

def normalize_date(value: str) -> str:
    v = str(value or "").strip().replace("-", "/")
    if re.fullmatch(r"\d{4}/\d{2}/\d{2}", v):
        y, m, d = v.split("/")
        return f"{y}-{m}-{d}"
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", v):
        d, m, y = v.split("/")
        return f"{y}-{m}-{d}"
    return ""

def clean_name(value: str) -> str:
    txt = re.sub(r"[^A-Za-z ]", " ", str(value or "").upper())
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def extract_fallback_fields(ocr_text: str) -> dict:
    pans = extract_pan_candidates_from_text(ocr_text)
    
    dob = ""
    for m in DOB_YMD_RE.findall(ocr_text):
        dob = normalize_date(m)
        if dob:
            break
    if not dob:
        for m in DOB_DMY_RE.findall(ocr_text):
            dob = normalize_date(m)
            if dob:
                break
    
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    lines_upper = [ln.upper() for ln in lines]
    
    name = ""
    father = ""
    
    for i, ln in enumerate(lines):
        if "NAME" in ln.upper() and i + 1 < len(lines):
            name = clean_name(lines[i + 1])
        if "FATHER" in ln.upper() and i + 1 < len(lines):
            father = clean_name(lines[i + 1])
    
    return {
        "pan_number": pans[0] if pans else "",
        "name": name,
        "fathers_name": father,
        "date_of_birth": dob,
    }

def merge_record_with_fallback(record: dict, ocr_text: str) -> dict:
    base = dict(record or {})
    fb = extract_fallback_fields(ocr_text)
    
    base["pan_number"] = normalize_pan_candidate(base.get("pan_number", "")) or fb.get("pan_number", "")
    base["name"] = clean_name(base.get("name", "")) or fb.get("name", "")
    base["fathers_name"] = clean_name(base.get("fathers_name", "")) or fb.get("fathers_name", "")
    base["date_of_birth"] = normalize_date(base.get("date_of_birth", "")) or fb.get("date_of_birth", "")
    return base

def _is_record_complete(record: dict) -> bool:
    """Fast completeness check to skip unnecessary LLM calls."""
    if not isinstance(record, dict):
        return False
    return all(
        bool(str(record.get(k, "")).strip())
        for k in ("pan_number", "name", "fathers_name", "date_of_birth")
    )

def extract_with_cohere(raw_text: str, co: cohere.Client) -> list[dict]:
    user_text = f"OCR TEXT:\n{raw_text}"
    
    def _chat_with_sdk_compat(model_name: str):
        try:
            return co.chat(
                model=model_name,
                system_prompt=SYSTEM_PROMPT,
                message=user_text,
                temperature=0.1,
            )
        except TypeError:
            try:
                return co.chat(
                    model=model_name,
                    preamble=SYSTEM_PROMPT,
                    message=user_text,
                    temperature=0.1,
                )
            except TypeError:
                return co.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=0.1,
                )
    
    response = None
    last_error = None
    
    try:
        response = _chat_with_sdk_compat(COHERE_MODEL)
    except Exception as e:
        last_error = e
        # Fallback to command-r if specified model fails
        try:
            response = _chat_with_sdk_compat("command-r-08-2024")
        except Exception as e2:
            last_error = e2
    
    if response is None:
        raise RuntimeError(f"Cohere failed: {last_error}")
    
    text = ""
    if hasattr(response, "text") and isinstance(response.text, str):
        text = response.text.strip()
    elif hasattr(response, "message") and hasattr(response.message, "content"):
        content = response.message.content
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif hasattr(item, "text") and isinstance(item.text, str):
                    parts.append(item.text)
            text = "".join(parts).strip()
    
    if not text:
        raise ValueError("Empty response from Cohere")
    
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    data = json.loads(text)
    return data if isinstance(data, list) else [data]

@st.cache_data(show_spinner=False)
def extract_with_cohere_cached(raw_text: str, model_name: str) -> list[dict]:
    """Cache LLM extraction by OCR text to speed reruns and duplicate pages."""
    co = cohere.Client(api_key=COHERE_API_KEY)
    return extract_with_cohere(raw_text, co)

def validate_pan(pan) -> bool:
    if not pan:
        return False
    return bool(PAN_RE.match(str(pan).strip().upper()))

# ── Main (optimized) ──────────────────────────────────────────────────────────
uploaded = st.file_uploader("📂 Upload PDF with PAN cards", type=["pdf"])

if uploaded:
    with st.spinner("🔄 Processing PDF..."):
        pdf_bytes = uploaded.read()
        try:
            pages = pdf_to_images_and_text(pdf_bytes)
        except Exception as e:
            st.error(f"PDF conversion failed: {e}")
            st.stop()
    
    st.success(f"✅ Loaded {len(pages)} page(s)")
    
    all_records: list[dict] = []
    page_results: list[dict] = []
    progress = st.progress(0.0)

    def process_page(i_page):
        i, page_img = i_page
        page_pdf_text = page_img.get("pdf_text", "")

        # Use embedded PDF text whenever possible; otherwise OCR image directly.
        if USE_PDF_TEXT_FIRST and is_pdf_text_usable(page_pdf_text):
            ocr_text = page_pdf_text
        else:
            ocr_text = _ocr_image_impl(page_img["image"])

        fallback_first = merge_record_with_fallback({}, ocr_text)
        if _is_record_complete(fallback_first):
            fallback_first["source_page"] = i + 1
            return i, [fallback_first], None

        try:
            records = extract_with_cohere_cached(ocr_text, COHERE_MODEL)
            merged_records = [merge_record_with_fallback(x, ocr_text) for x in records]
            if not merged_records:
                merged_records = [merge_record_with_fallback({}, ocr_text)]
            for r in merged_records:
                r["source_page"] = i + 1
            return i, merged_records, None
        except Exception as e:
            fallback = merge_record_with_fallback({}, ocr_text)
            fallback["source_page"] = i + 1
            return i, [fallback], str(e)

    max_workers = min(3, len(pages)) if pages else 1
    completed = 0
    ordered_results = [None] * len(pages)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {executor.submit(process_page, x) for x in enumerate(pages)}
        displayed_progress = 0.0

        while pending:
            done, pending = wait(pending, timeout=0.12, return_when=FIRST_COMPLETED)

            for fut in done:
                i, records, err = fut.result()
                ordered_results[i] = {"page": i + 1, "records": records, "error": err}
                completed += 1

            target_progress = completed / len(pages)

            # Smooth progress updates so the bar visibly advances during work.
            while displayed_progress + 1e-9 < target_progress:
                displayed_progress = min(target_progress, displayed_progress + 0.02)
                progress.progress(displayed_progress, text=f"Processed {completed}/{len(pages)} pages")
                time.sleep(0.015)

            # Keep status alive even between completions.
            if not done:
                progress.progress(displayed_progress, text=f"Processing... {completed}/{len(pages)} pages done")

    for pr in ordered_results:
        if pr is None:
            continue
        page_results.append(pr)
        all_records.extend(pr["records"])

    progress.empty()
    
    # Metrics
    valid_count = sum(1 for r in all_records if validate_pan(r.get("pan_number")))
    cols = st.columns(4)
    metrics = [len(pages), len(all_records), valid_count, len(all_records)-valid_count]
    labels = ["Pages", "Cards Found", "Valid PANs", "Needs Review"]
    for col, num, lbl in zip(cols, metrics, labels):
        col.markdown(f"""<div class="metric-box"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>""", 
                    unsafe_allow_html=True)
    
    # Results table
    st.markdown("## 📋 Extracted Records")
    if all_records:
        df = pd.DataFrame(all_records)
        fixed_cols = ["source_page", "pan_number", "name", "fathers_name", "date_of_birth"]
        for c in fixed_cols:
            if c not in df.columns:
                df[c] = None
        df = df[fixed_cols + [c for c in df.columns if c not in fixed_cols]]
        df["pan_valid"] = df["pan_number"].apply(lambda x: "✅ Valid" if validate_pan(x) else "⚠️ Check")
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_buf.getvalue(),
            file_name="pan_cards_extracted.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No PAN card data extracted.")
    
    # Page details (minimal)
    with st.expander("🔍 Page Details"):
        for pr in page_results:
            st.markdown(f"**Page {pr['page']}** — {len(pr['records'])} card(s)")
            if pr["error"]:
                st.caption(f"Error: {pr['error']}")
else:
    st.markdown("""
    <div class="card" style="text-align:center;padding:2rem;">
        <div style="font-size:3rem;">📄</div>
        <h3 style="color:#1a237e;">Upload a PDF to get started</h3>
    </div>
    """, unsafe_allow_html=True)
