"""
Vision Transformer (ViT): Interactive Implementation Walkthrough
SEAS 8525: Computer Vision and Generative AI
Dr. Elbasheer

Run:
    pip install streamlit torch torchvision transformers einops matplotlib pandas seaborn pillow opencv-python-headless
    streamlit run vit_app.py

On Google Colab:
    !pip install streamlit pyngrok torch torchvision transformers einops matplotlib pandas seaborn pillow opencv-python-headless
    !streamlit run vit_app.py &
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print(public_url)
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from torchvision import transforms
import math

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ViT Implementation Walkthrough",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
.main { padding-top: 0; }
.block-container { padding-top: 1.4rem; max-width: 1160px; }

/* ── Hero ── */
.hero {
    padding: 2.6rem 0 2rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.8rem;
}
.hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: #4f46e5;
    margin: 0 0 0.8rem;
}
.hero h1 {
    font-family: 'Inter', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.14;
    color: #0f172a;
    margin: 0 0 0.7rem;
}
.hero-body {
    font-family: 'Inter', sans-serif;
    font-size: 0.94rem;
    color: #475569;
    margin: 0;
    line-height: 1.8;
    max-width: 620px;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin: 0 0 2rem;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #4f46e5;
    border-radius: 0 0 8px 8px;
    padding: 16px 12px 14px;
    text-align: center;
}
.metric-val {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1;
}
.metric-lbl {
    display: block;
    font-family: 'Inter', sans-serif;
    font-size: 0.73rem;
    font-weight: 600;
    color: #374151;
    margin-top: 5px;
}
.metric-desc {
    display: block;
    font-family: 'Inter', sans-serif;
    font-size: 0.66rem;
    color: #94a3b8;
    margin-top: 3px;
    line-height: 1.4;
}

/* ── Section header ── */
.section-header {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 0 0 1rem;
    border-bottom: 2px solid #e2e8f0;
    margin: 2.2rem 0 1.5rem;
}
.section-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #ffffff;
    background: #4f46e5;
    border-radius: 5px;
    padding: 3px 9px;
    flex-shrink: 0;
    margin-top: 5px;
    line-height: 1.5;
}
.section-header h2 {
    font-family: 'Inter', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #0f172a;
    margin: 0 0 4px;
    line-height: 1.2;
}
.section-header p {
    font-family: 'Inter', sans-serif;
    font-size: 0.84rem;
    color: #64748b;
    margin: 0;
}

/* ── Callout boxes ── */
.box-label {
    display: block;
    font-family: 'Inter', sans-serif;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.info-box {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 8px;
    padding: 14px 17px;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    margin: 1rem 0;
    line-height: 1.72;
    color: #1e1b4b;
}
.info-box .box-label { color: #4338ca; }

.success-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 14px 17px;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    margin: 1rem 0;
    line-height: 1.72;
    color: #14532d;
}
.success-box .box-label { color: #15803d; }

.warn-box {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 14px 17px;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    margin: 1rem 0;
    line-height: 1.72;
    color: #78350f;
}
.warn-box .box-label { color: #b45309; }

/* ── Inline tensor shape badge ── */
.dim-badge {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 1px 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #334155;
    margin: 0 2px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] > div:first-child { background: #eef2ff !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label span { color: #374151 !important; font-size: 0.84rem; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] hr { border-color: #c7d2fe; }

.sidebar-brand {
    padding: 2px 0 14px;
    border-bottom: 1px solid #c7d2fe;
    margin-bottom: 4px;
}
.sidebar-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.97rem;
    font-weight: 700;
    color: #1e1b4b;
    letter-spacing: -0.01em;
}
.sidebar-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #6366f1;
    font-weight: 500;
    margin-top: 3px;
}
.sidebar-setup {
    background: #e0e7ff;
    border: 1px solid #c7d2fe;
    border-radius: 8px;
    padding: 12px 14px;
}
.sidebar-setup-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #4338ca;
    margin-bottom: 7px;
}
.setup-block { margin-bottom: 11px; }
.setup-block:last-child { margin-bottom: 0; }
.setup-tag {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #fff;
    padding: 2px 9px;
    border-radius: 3px;
    margin-bottom: 6px;
}
.setup-tag.conda { background: #4338ca; }
.setup-tag.pip   { background: #7c3aed; }
.setup-tag .rec  { font-weight: 400; font-style: italic; opacity: 0.88; margin-left: 2px; }
.setup-cmd {
    display: block;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 0.72rem;
    color: #1e1b4b;
    background: #ffffff;
    border: 1px solid #c7d2fe;
    border-radius: 5px;
    padding: 5px 10px;
    margin-bottom: 4px;
    line-height: 1.5;
    word-break: break-all;
}
.setup-cmd:last-child { margin-bottom: 0; }
.device-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #e0e7ff;
    border: 1px solid #c7d2fe;
    border-radius: 7px;
    padding: 9px 13px;
    font-family: 'Inter', sans-serif;
    font-size: 0.80rem;
    width: 100%;
    margin-top: 4px;
}
.device-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #4f46e5 !important;
    border: none !important;
    border-radius: 7px;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.875rem;
    padding: 0.45rem 1.2rem;
}
.stButton > button[kind="primary"]:hover { background: #4338ca !important; }

.bridge-same { color: #059669; font-weight: 700; }
.bridge-diff { color: #d97706; font-weight: 700; }

/* ── Sidebar brand extras ── */
.sidebar-course { font-family:'Inter',sans-serif; font-size:0.90rem; font-weight:700; color:#3730a3; margin-top:8px; }
.sidebar-course-name { font-family:'Inter',sans-serif; font-size:0.83rem; font-weight:600; color:#4f46e5; margin-top:2px; }
.sidebar-instructor { font-family:'Inter',sans-serif; font-size:0.77rem; color:#6366f1; margin-top:4px; font-style:italic; }

/* ── Comparison table ── */
.comp-table { width:100%; border-collapse:collapse; font-family:'Inter',sans-serif; font-size:0.875rem; margin:0.5rem 0 1.2rem; }
.comp-table thead th { background:#f8fafc; color:#374151; font-weight:700; font-size:0.70rem; letter-spacing:0.07em; text-transform:uppercase; padding:11px 16px; text-align:left; border-bottom:2px solid #e2e8f0; }
.comp-table tbody td { padding:10px 16px; color:#1e293b; border-bottom:1px solid #f1f5f9; vertical-align:top; line-height:1.55; }
.comp-table tbody tr:last-child td { border-bottom:none; }
.comp-table tbody tr:hover td { background:#fafbff; }
.comp-table td:first-child { font-weight:600; color:#374151; white-space:nowrap; width:18%; }
.comp-table th:last-child, .comp-table td:last-child { text-align:center; width:11%; }
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.68rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; white-space:nowrap; }
.badge-changed   { background:#fef3c7; color:#b45309; border:1px solid #fde68a; }
.badge-identical { background:#d1fae5; color:#065f46; border:1px solid #a7f3d0; }
.badge-sameidea  { background:#dbeafe; color:#1d4ed8; border:1px solid #bfdbfe; }

/* ── Architecture comparison ── */
.arch-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:0.5rem 0 1.2rem; }
.arch-card { border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }
.arch-card-hdr { padding:11px 16px; font-family:'Inter',sans-serif; font-size:0.82rem; font-weight:700; color:#fff; letter-spacing:0.02em; }
.arch-card-hdr.nlp { background:#4f46e5; }
.arch-card-hdr.vit { background:#059669; }
.arch-card-body { padding:14px 16px; background:#ffffff; }
.arch-row { display:flex; align-items:flex-start; gap:10px; margin-bottom:2px; }
.arch-num { background:#f1f5f9; border:1px solid #e2e8f0; border-radius:4px; font-family:'JetBrains Mono',monospace; font-size:0.66rem; font-weight:700; color:#4f46e5; padding:2px 6px; flex-shrink:0; margin-top:2px; }
.arch-text { font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#1e293b; line-height:1.6; }
.arch-arrow { color:#c7d2fe; font-size:0.80rem; margin:1px 0 1px 36px; font-weight:700; }

/* ── Encoder block component cards ── */
.enc-cards { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:14px 0 20px; }
.enc-card { border-radius:9px; padding:14px 16px; }
.enc-card-title { font-family:'Inter',sans-serif; font-size:0.88rem; font-weight:700; margin-bottom:7px; }
.enc-card-body { font-family:'Inter',sans-serif; font-size:0.82rem; color:#374151; line-height:1.68; }
.enc-card-formula { font-family:'JetBrains Mono','Courier New',monospace; font-size:0.77rem; background:rgba(0,0,0,0.04); border-radius:4px; padding:4px 8px; margin-top:7px; display:block; color:#1e293b; }

/* ── Pre/post-norm comparison ── */
.norm-compare { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:14px 0 20px; }
.norm-col { display:flex; flex-direction:column; align-items:center; }
.norm-header { font-family:'Inter',sans-serif; font-size:0.84rem; font-weight:700; padding:7px 14px; border-radius:6px; margin-bottom:10px; text-align:center; width:92%; box-sizing:border-box; }
.norm-header.post { background:#fef3c7; color:#b45309; border:1px solid #fde68a; }
.norm-header.pre  { background:#dcfce7; color:#14532d; border:1px solid #bbf7d0; }
.norm-row { width:88%; margin-bottom:3px; }
.nb { padding:8px 10px; border-radius:6px; text-align:center; font-family:'Inter',sans-serif; font-size:0.78rem; font-weight:600; width:100%; box-sizing:border-box; }
.nb-input  { background:#eef2ff; border:2px solid #4f46e5; color:#3730a3; }
.nb-attn   { background:#fdf2f8; border:2px solid #db2777; color:#9d174d; }
.nb-ffn    { background:#fffbeb; border:2px solid #d97706; color:#92400e; }
.nb-resid  { background:#f8fafc; border:2px solid #94a3b8; color:#64748b; font-size:0.72rem; }
.nb-ln-post { background:#fee2e2; border:2px solid #dc2626; color:#991b1b; }
.nb-ln-pre  { background:#dcfce7; border:2px solid #16a34a; color:#14532d; }
.nb-output { background:#eef2ff; border:2px solid #4f46e5; color:#3730a3; }
.norm-arrow { color:#6366f1; font-size:1.1rem; text-align:center; width:88%; margin:2px 0; font-weight:700; }
.norm-note { font-family:'Inter',sans-serif; font-size:0.70rem; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:4px; padding:5px 10px; margin-top:8px; color:#14532d; width:92%; box-sizing:border-box; text-align:center; }
.postnorm-note { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; }

/* ── Q/K/V cards ── */
.qkv-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin:14px 0 18px; }
.qkv-card { border-radius:9px; padding:14px 16px; border:2px solid transparent; }
.qkv-card.q-card { background:#ede9fe; border-color:#7c3aed; }
.qkv-card.k-card { background:#dbeafe; border-color:#2563eb; }
.qkv-card.v-card { background:#d1fae5; border-color:#059669; }
.qkv-title { font-family:'Inter',sans-serif; font-size:0.90rem; font-weight:700; margin-bottom:6px; }
.qkv-card.q-card .qkv-title { color:#6d28d9; }
.qkv-card.k-card .qkv-title { color:#1d4ed8; }
.qkv-card.v-card .qkv-title { color:#047857; }
.qkv-body { font-family:'Inter',sans-serif; font-size:0.80rem; color:#374151; line-height:1.65; }
.qkv-em { font-style:italic; font-weight:600; }

/* ── Pretrained explainer card ── */
.pretrained-card { background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #4f46e5; border-radius:8px; padding:16px 20px; margin:14px 0 18px; }
.pretrained-card h4 { font-family:'Inter',sans-serif; font-size:0.90rem; font-weight:700; color:#1e1b4b; margin:0 0 10px; }
.pretrained-row { display:flex; gap:10px; margin-bottom:8px; align-items:flex-start; }
.pretrained-icon { font-size:1.0rem; flex-shrink:0; margin-top:1px; }
.pretrained-text { font-family:'Inter',sans-serif; font-size:0.845rem; color:#374151; line-height:1.65; }
.pretrained-text strong { color:#1e293b; }

/* ── Patch embedding pipeline visual ── */
.pipeline { display:flex; align-items:flex-start; gap:4px; margin:14px 0 18px; flex-wrap:wrap; }
.pipe-step { display:flex; flex-direction:column; align-items:center; gap:5px; }
.pipe-box { background:#f8fafc; border:2px solid #c7d2fe; border-radius:8px; padding:9px 12px; text-align:center; font-family:'Inter',sans-serif; font-size:0.82rem; font-weight:700; color:#1e1b4b; min-width:96px; }
.pipe-box small { display:block; font-size:0.69rem; font-weight:400; color:#6366f1; margin-top:3px; line-height:1.4; }
.pipe-label { font-family:'Inter',sans-serif; font-size:0.67rem; color:#64748b; text-align:center; max-width:96px; line-height:1.3; }
.pipe-arrow { font-size:1.4rem; color:#4f46e5; font-weight:700; margin-top:22px; padding:0 2px; }
.math-box { background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #4f46e5; border-radius:6px; padding:14px 18px; margin:12px 0; font-family:'Inter',sans-serif; font-size:0.875rem; color:#1e293b; line-height:1.75; }
.math-box code { background:#eef2ff; border:1px solid #c7d2fe; border-radius:3px; padding:1px 6px; font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#3730a3; }
.math-step { display:flex; gap:10px; margin-bottom:8px; align-items:flex-start; }
.math-step-num { background:#4f46e5; color:#fff; border-radius:50%; width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-size:0.70rem; font-weight:700; flex-shrink:0; margin-top:1px; font-family:'JetBrains Mono',monospace; }
.math-step-text { font-family:'Inter',sans-serif; font-size:0.855rem; color:#1e293b; line-height:1.65; }

/* ── Q&A practice ── */
.qa-item { margin:0 0 14px; border:1px solid #e2e8f0; border-radius:8px; overflow:hidden; }
.qa-q { background:#f8fafc; padding:11px 16px; font-family:'Inter',sans-serif; font-size:0.875rem; font-weight:600; color:#0f172a; border-bottom:1px solid #e2e8f0; }
.qa-num { display:inline-block; background:#4f46e5; color:#fff; border-radius:4px; font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:700; padding:1px 7px; margin-right:9px; }
.qa-a { background:#ffffff; padding:11px 16px; font-family:'Inter',sans-serif; font-size:0.855rem; color:#374151; line-height:1.75; }
.qa-a strong { color:#1e293b; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ──────────────────────────────────────────────────────────────────
def section_header(num, title, subtitle=""):
    num_str = str(num).zfill(2)
    sub_html = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="section-header">
        <span class="section-num">{num_str}</span>
        <div>
            <h2>{title}</h2>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def info(text):
    st.markdown(
        f'<div class="info-box"><span class="box-label">Class Tip</span>{text}</div>',
        unsafe_allow_html=True)

def success(text):
    st.markdown(
        f'<div class="success-box"><span class="box-label">Summary</span>{text}</div>',
        unsafe_allow_html=True)

def warn(text):
    st.markdown(
        f'<div class="warn-box"><span class="box-label">Important</span>{text}</div>',
        unsafe_allow_html=True)

def metric_cards(items):
    """items: list of (value, label, description) tuples"""
    html = "".join(
        f'<div class="metric-card">'
        f'<span class="metric-val">{v}</span>'
        f'<span class="metric-lbl">{l}</span>'
        f'<span class="metric-desc">{d}</span>'
        f'</div>'
        for v, l, d in items
    )
    st.markdown(f'<div class="metric-row">{html}</div>', unsafe_allow_html=True)

@st.cache_resource
def load_vit_model():
    """Load pretrained ViT-Base/16 from HuggingFace (cached after first load)."""
    from transformers import ViTModel, ViTImageProcessor
    model    = ViTModel.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        output_attentions=True
    )
    processor = ViTImageProcessor.from_pretrained(
        'google/vit-base-patch16-224-in21k'
    )
    model.eval()
    return model, processor

def preprocess_image(img: Image.Image, size=224) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])
    return transform(img).unsqueeze(0)  # (1, 3, 224, 224)

def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """Denormalize tensor back to displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img  = t.squeeze(0) * std + mean
    img  = img.clamp(0,1).permute(1,2,0).numpy()
    return img


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-title">ViT Walkthrough</div>
        <div class="sidebar-course">SEAS 8525</div>
        <div class="sidebar-course-name">Computer Vision &amp; Generative AI</div>
        <div class="sidebar-instructor">Dr. Elbasheer</div>
    </div>
    """, unsafe_allow_html=True)

    section = st.radio(
        "Section",
        options=[
            "1 · The Bridge",
            "2 · Patch Extraction",
            "3 · Patch Embedding",
            "4 · CLS Token + Position",
            "5 · Self-Attention",
            "6 · Encoder Block",
            "7 · MLP Head",
            "8 · Full Model",
            "9 · Training",
        ]
    )
    st.divider()

    st.markdown("**Upload an image**")
    uploaded = st.file_uploader(
        "Used in sections 1, 2, 4",
        type=["png","jpg","jpeg","webp"],
        label_visibility="visible"
    )
    if uploaded:
        user_img = Image.open(uploaded).convert("RGB")
        st.image(user_img, caption="Our image", use_container_width=True)
    else:
        # Generate a simple default test image
        _i, _j = np.meshgrid(np.arange(224), np.arange(224), indexing='ij')
        arr = np.stack([
            (255*(0.5+0.5*np.sin(_i/30))).astype(np.uint8),
            (255*(0.5+0.5*np.cos(_j/25))).astype(np.uint8),
            (255*(0.5+0.5*np.sin((_i+_j)/40))).astype(np.uint8),
        ], axis=-1)
        user_img = Image.fromarray(arr)
        st.image(user_img, caption="Default test image (upload yours)", use_container_width=True)

    st.divider()
    st.markdown("""
    <div class="sidebar-setup">
        <div class="sidebar-setup-label">Quick setup</div>
        <div class="setup-block">
            <div class="setup-tag conda">Conda <span class="rec">recommended</span></div>
            <div class="setup-cmd">conda env create -f environment.yml</div>
            <div class="setup-cmd">conda activate vit-walkthrough</div>
            <div class="setup-cmd">streamlit run vit_app.py</div>
        </div>
        <div class="setup-block">
            <div class="setup-tag pip">pip</div>
            <div class="setup-cmd">pip install -r requirements.txt</div>
            <div class="setup-cmd">streamlit run vit_app.py</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        st.markdown(f"""
        <div class="device-badge">
            <span class="device-dot" style="background:#10b981;"></span>
            <span style="color:#065f46;font-weight:700;">GPU</span>
            <span style="color:#374151;">{torch.cuda.get_device_name(0)}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="device-badge">
            <span class="device-dot" style="background:#f59e0b;"></span>
            <span style="color:#92400e;font-weight:700;">CPU</span>
            <span style="color:#4b5563;">GPU strongly recommended; ViT self-attention is slow on CPU</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0: THE BRIDGE
# ══════════════════════════════════════════════════════════════════════════════
if section.startswith("1"):
    st.markdown("""
    <div class="hero">
        <p class="hero-eyebrow">SEAS 8525 &nbsp;&middot;&nbsp; Computer Vision and Generative AI</p>
        <h1>Vision Transformer</h1>
        <p class="hero-body">
            Dosovitskiy et al. (2020) &nbsp;&middot;&nbsp;
            <em>"An Image is Worth 16&times;16 Words"</em><br>
            An interactive implementation walkthrough: from raw pixels to image classification.
            Each section builds one component of ViT-Base, the reference model with the numbers below.
        </p>
    </div>
    """, unsafe_allow_html=True)

    metric_cards([
        ("86M",  "Parameters",     "total learnable weights in ViT-Base"),
        ("12",   "Encoder Layers", "transformer blocks stacked in sequence"),
        ("196",  "Patch Tokens",   "patches per image at 16×16 px each"),
        ("768",  "Embed Dim",      "vector size representing each token"),
        ("12",   "Attn Heads",     "parallel attention per encoder layer"),
    ])

    section_header("1", "The Bridge", "Patches replace words; the encoder stays the same")

    info("""
    ViT does not invent a new architecture.
    It asks one question: <em>what if we treated image patches as tokens?</em>
    The transformer encoder is used completely unchanged.
    """)

    st.markdown("### What changes vs what stays the same")

    _rows = [
        ("Input unit",         "Word token",                       "16&times;16 pixel patch",             "changed"),
        ("Tokenization",       "Vocabulary lookup (integer ID)",   "Flatten patch + linear projection",   "changed"),
        ("Embedding dim",      "512 (lookup table)",               "768 (linear layer W<sub>E</sub>)",    "changed"),
        ("Sequence length",    "Number of words",                  "(H/P)(W/P) patches + 1 CLS",          "changed"),
        ("CLS token",          "Sometimes (BERT)",                 "Always prepended",                    "changed"),
        ("Positional encoding","Sinusoidal or learned",            "Learned 1D over patch positions",     "sameidea"),
        ("Self-attention",     "Multi-head Q/K/V",                 "Multi-head Q/K/V (unchanged)",        "identical"),
        ("Feed-forward (FFN)", "2-layer MLP per position",         "2-layer MLP (unchanged)",             "identical"),
        ("Add &amp; Norm",     "Residual + LayerNorm",             "Residual + LayerNorm (unchanged)",    "identical"),
        ("Output head",        "Linear over last token",           "Linear over CLS token only",          "changed"),
    ]
    _badge_label = {"changed":"Changed", "identical":"Identical", "sameidea":"Same idea"}
    _rows_html = ""
    for comp, nlp, vit, status in _rows:
        _rows_html += (
            f'<tr>'
            f'<td>{comp}</td>'
            f'<td>{nlp}</td>'
            f'<td>{vit}</td>'
            f'<td><span class="badge badge-{status}">{_badge_label[status]}</span></td>'
            f'</tr>'
        )
    st.markdown(f"""
    <table class="comp-table">
      <thead>
        <tr>
          <th>Component</th>
          <th>Transformer (NLP)</th>
          <th>ViT (Vision)</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{_rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Architecture comparison")

    st.markdown("""
    <div class="arch-grid">
      <div class="arch-card">
        <div class="arch-card-hdr nlp">Transformer (NLP)</div>
        <div class="arch-card-body">
          <div class="arch-row"><span class="arch-num">1</span><span class="arch-text">Sentence: "The cat sat on mat"</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">2</span><span class="arch-text">Tokenize &#8594; [4, 831, 2756, 15, 9104]</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">3</span><span class="arch-text">Embedding lookup &#8594; (5, 512)</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">4</span><span class="arch-text">+ Sinusoidal positional encoding</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">5</span><span class="arch-text">Transformer encoder &times; 6</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">6</span><span class="arch-text">Linear + Softmax &#8594; next token</span></div>
        </div>
      </div>
      <div class="arch-card">
        <div class="arch-card-hdr vit">Vision Transformer (ViT)</div>
        <div class="arch-card-body">
          <div class="arch-row"><span class="arch-num">1</span><span class="arch-text">Image: (3, 224, 224)</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">2</span><span class="arch-text">Split into 16&times;16 patches &#8594; 196 patches</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">3</span><span class="arch-text">Flatten + Linear project &#8594; (196, 768)</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">4</span><span class="arch-text">Prepend CLS &#8594; (197, 768) + learned pos. enc.</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">5</span><span class="arch-text">Transformer encoder &times; 12</span></div>
          <div class="arch-arrow">&#8595;</div>
          <div class="arch-row"><span class="arch-num">6</span><span class="arch-text">CLS output &#8594; Linear &#8594; class label</span></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    warn("""
    <strong>Why does ViT need large data?</strong>
    CNNs have built-in inductive biases: <em>locality</em> (nearby pixels relate)
    and <em>translation equivariance</em> (a cat is a cat anywhere in the image).
    ViT has neither; it must learn these spatial relationships from data.
    On small datasets, CNNs win. On ImageNet-21k or JFT-300M, ViT surpasses CNNs.
    """)

    with st.expander("Practice Questions: ViT vs CNN (Midterm Prep)", expanded=False):
        st.markdown("""
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q1</span>What inductive biases do CNNs have that ViT lacks?</div>
          <div class="qa-a">CNNs have two built-in inductive biases: <strong>locality</strong> (conv filters operate on small neighborhoods, assuming nearby pixels are related) and <strong>translation equivariance</strong> (the same filter fires regardless of where a feature appears). ViT uses global self-attention, so it treats every pair of patches equally and must learn spatial structure from data. This is why ViT needs large-scale pre-training.</div>
        </div>
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q2</span>Why does ViT outperform CNNs on large datasets but not on small ones?</div>
          <div class="qa-a">Without inductive biases, ViT must learn from scratch that nearby patches tend to be related. That requires many examples. CNNs generalize well from small data precisely because locality and equivariance are already wired in. At scale (ImageNet-21k: 14M images; JFT-300M: 300M images) ViT overtakes CNNs because it can exploit global context that convolutions miss.</div>
        </div>
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q3</span>What replaces the word-embedding lookup table in ViT?</div>
          <div class="qa-a">A linear projection (implemented as a Conv2d with kernel size = stride = P = 16). Each 16&times;16 patch is flattened to a 768-dim vector and multiplied by a learned weight matrix W<sub>E</sub> &#8712; &#8477;<sup>768&times;768</sup>. The Conv2d processes all patches in one GPU pass, making it far faster than looping over patches.</div>
        </div>
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q4</span>What is the CLS token and why is it needed?</div>
          <div class="qa-a">A learnable parameter vector prepended to the patch sequence, giving a (197, 768) input to the encoder. It acts as a global summary token: through 12 layers of self-attention it aggregates information from all 196 patches. Only its final output is fed to the linear classification head. This design keeps the encoder architecture identical to NLP transformers.</div>
        </div>
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q5</span>What is the computational complexity of self-attention, and why do patches help?</div>
          <div class="qa-a">Self-attention is <strong>O(N&sup2;)</strong> in sequence length N. Processing every pixel: N = 224&sup2; = 50,176 &rarr; 50,176&sup2; &asymp; 2.5&thinsp;billion attention ops per image. Patches reduce N to 196 &rarr; 196&sup2; = 38,416 ops, roughly <strong>65,000&times; fewer</strong>. This makes the transformer tractable for vision.</div>
        </div>
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q6</span>How does positional encoding differ between NLP Transformers and ViT?</div>
          <div class="qa-a">NLP transformers typically use fixed <em>sinusoidal</em> positional encodings. ViT uses <em>learned 1D</em> positional embeddings, one per patch position (0 to 196). Despite being 1D, the learned embeddings develop 2D spatial structure on their own. At fine-tuning time on different resolutions the positional embeddings are interpolated.</div>
        </div>
        <div class="qa-item">
          <div class="qa-q"><span class="qa-num">Q7</span>What is &ldquo;attention rollout&rdquo; and what does it reveal?</div>
          <div class="qa-a">A visualization technique that propagates attention through all 12 encoder layers by recursively multiplying averaged attention matrices. The result shows which input patches most influence the CLS token&rsquo;s final representation. Rollout maps often highlight semantically meaningful regions (e.g., the foreground object) even without segmentation supervision.</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PATCH EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("2"):
    section_header("2", "Patch Extraction", "Slicing an image into a sequence of fixed-size patches")

    # Pre-compute image once so all sub-sections share the same object
    img_resized = user_img.resize((224, 224))
    img_tensor  = preprocess_image(img_resized)

    # ── Zone 1: Concept + parameters ─────────────────────────────
    col_concept, col_params = st.columns([1.3, 1])

    with col_concept:
        st.markdown("### Concept")
        st.markdown("""
        An image of size **H × W × C** is divided into non-overlapping patches of size **P × P**.
        This gives us **N = (H/P)(W/P)** patches.

        For a standard 224×224 image with P = 16:
        - Grid: 14 × 14 = **196 patches**
        - Each patch flattened: 16×16×3 = **768 dimensions**
        """)
        info("""
        <strong>Why patches?</strong> Every pixel as a token gives 224² = 50,176 tokens.
        Self-attention is O(N²), so that is 50,176² ≈ 2.5 billion pairs per image.
        Patches shrink this to 196² = 38,416, about 65,000× fewer operations.
        """)

    with col_params:
        st.markdown("### Parameters")
        patch_size = st.select_slider(
            "Patch size P",
            options=[8, 16, 32],
            value=16
        )
        n_patches_side = 224 // patch_size
        n_patches      = n_patches_side ** 2
        patch_vec_size = patch_size * patch_size * 3
        st.markdown(f"""
        **With P = {patch_size}:**
        - Grid: {n_patches_side} × {n_patches_side}
        - Total patches: **{n_patches}**
        - Flattened size: {patch_size}×{patch_size}×3 = **{patch_vec_size}**
        - Sequence shape: **({n_patches}, 768)** after embedding
        """)

    # ── Zone 2: Thumbnail + Extract button ───────────────────────
    st.markdown("### Try it on our image")
    thumb_col, btn_col = st.columns([1, 3])

    with thumb_col:
        st.image(img_resized, caption="Input image", use_container_width=True)

    with btn_col:
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.875rem;"
            "color:#374151;margin-bottom:12px;'>"
            "Click to split the image into patches using the patch size above. "
            "Results appear below.</p>",
            unsafe_allow_html=True
        )
        if st.button("Extract patches", type="primary", key="extract_btn"):
            with st.spinner("Extracting..."):
                from einops import rearrange
                patches = rearrange(
                    img_tensor,
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_size, p2=patch_size
                )
                st.success(f"patches.shape = {tuple(patches.shape)}")

                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                fig.patch.set_facecolor('white')
                img_np = np.array(img_resized)
                axes[0].imshow(img_np)
                axes[0].set_title("Original", color='#0f172a', fontsize=10)
                axes[0].axis('off')
                axes[1].imshow(img_np)
                for i in range(n_patches_side + 1):
                    axes[1].axhline(i * patch_size, color='#7f77dd', lw=0.8, alpha=0.7)
                    axes[1].axvline(i * patch_size, color='#7f77dd', lw=0.8, alpha=0.7)
                axes[1].set_title(f"Patch grid ({patch_size}×{patch_size})", color='#0f172a', fontsize=10)
                axes[1].axis('off')
                for ax in axes:
                    ax.set_facecolor('white')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("**First 6 extracted patches:**")
                fig2, axes2 = plt.subplots(1, 6, figsize=(10, 2.5))
                fig2.patch.set_facecolor('white')
                img_arr = np.array(img_resized)
                for idx in range(6):
                    r = idx // n_patches_side
                    c = idx  % n_patches_side
                    patch = img_arr[r*patch_size:(r+1)*patch_size,
                                    c*patch_size:(c+1)*patch_size]
                    axes2[idx].imshow(patch)
                    axes2[idx].set_title(f"P-{idx+1}", color='#0f172a', fontsize=9)
                    axes2[idx].axis('off')
                    axes2[idx].set_facecolor('white')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

                st.markdown("**Patch tensor values (patch 1, first 16 dims):**")
                flat = patches[0, 0, :16].detach().numpy()
                st.code(f"patches[0, 0, :16] =\n{np.round(flat, 4)}")

    # ── Zone 3: PyTorch implementation ───────────────────────────
    st.divider()
    st.markdown("### PyTorch implementation")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:8px;'>Here is the code that performs the extraction above. "
        "Two approaches: <code>einops</code> (readable) and <code>unfold</code> "
        "(standard PyTorch):</p>",
        unsafe_allow_html=True
    )
    st.code("""
import torch
from einops import rearrange

def extract_patches(x, patch_size=16):
    # x: (B, C, H, W) = (1, 3, 224, 224)
    B, C, H, W = x.shape
    P = patch_size
    # Rearrange: (B, C, H, W) -> (B, N, P*P*C)
    patches = rearrange(
        x,
        'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
        p1=P, p2=P
    )
    # patches.shape = (B, 196, 768)
    return patches


# Equivalent without einops (standard PyTorch unfold):
def extract_patches_unfold(x, patch_size=16):
    B, C, H, W = x.shape
    P = patch_size
    x = x.unfold(2, P, P).unfold(3, P, P)
    # shape: (B, C, n_h, n_w, P, P)
    x = x.contiguous().view(B, C, -1, P*P)
    x = x.permute(0, 2, 3, 1).flatten(2)
    # shape: (B, 196, 768)
    return x
    """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PATCH EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("3"):
    section_header("3", "Patch Embedding", "Turning raw pixel patches into vectors the transformer can read")

    img_resized = user_img.resize((224, 224))
    img_tensor  = preprocess_image(img_resized)

    # ── Zone 1: What is embedding? ───────────────────────────────
    st.markdown("### What is an embedding?")
    st.markdown("""
    An **embedding** is just a list of numbers that represents something in a way a
    neural network can process. Think of it like a fingerprint for a patch: two patches
    with similar colors and textures will produce similar embeddings.

    After Section 2 we have **196 patches**, each a raw 16×16×3 block of pixel values.
    Patch embedding converts each of those raw patches into a **768-number vector**
    that the transformer encoder can actually work with.
    """)

    # Visual pipeline
    st.markdown("### How it works: step by step")
    st.markdown("""
    <div class="pipeline">
      <div class="pipe-step">
        <div class="pipe-box">16×16×3 patch<small>one tile from the image<br>768 raw pixel numbers</small></div>
        <div class="pipe-label">Start here</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-step">
        <div class="pipe-box">Flatten<small>stack all pixels<br>into one long row</small></div>
        <div class="pipe-label">Reshape</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-step">
        <div class="pipe-box">vector (768,)<small>768 numbers<br>in a single row</small></div>
        <div class="pipe-label">1-D vector</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-step">
        <div class="pipe-box">&#215; W<sub>E</sub><small>multiply by<br>learned weights</small></div>
        <div class="pipe-label">Linear projection</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-step">
        <div class="pipe-box">embedding (768,)<small>rich representation<br>ready for transformer</small></div>
        <div class="pipe-label">Done &#10003;</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Plain-English math
    st.markdown("### The math in plain English")
    st.markdown("""
    <div class="math-box">
      <div class="math-step">
        <div class="math-step-num">1</div>
        <div class="math-step-text">
          Take one patch: a 16&times;16 grid with 3 color channels (R, G, B).
          That gives <code>16 &times; 16 &times; 3 = 768</code> raw pixel numbers.
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">2</div>
        <div class="math-step-text">
          <strong>Flatten</strong> it: lay all 768 numbers in a single row to get a vector of shape <code>(768,)</code>.
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">3</div>
        <div class="math-step-text">
          <strong>Multiply</strong> that vector by the weight matrix W<sub>E</sub> of shape <code>(768, 768)</code>.
          Think of W<sub>E</sub> as a 768&times;768 grid of learnable numbers.
          The result is a new vector of shape <code>(768,)</code>: the patch embedding.
        </div>
      </div>
      <div class="math-step">
        <div class="math-step-num">4</div>
        <div class="math-step-text">
          Repeat for all 196 patches. The <strong>same</strong> W<sub>E</sub> is applied to every patch.
          Final output: <code>(196, 768)</code>: 196 patch embeddings, each of length 768.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#1e293b;margin:10px 0 16px;'>"
        "<strong>How many parameters does W<sub>E</sub> add?</strong><br>"
        "768 &times; 768 = <strong>589,824 learnable numbers</strong> just for this one layer. "
        "Training adjusts them so patches with similar content end up with similar vectors.</p>",
        unsafe_allow_html=True
    )

    info("""
    <strong>Why use Conv2d instead of a loop?</strong><br><br>
    The obvious approach: loop over all 196 patches, multiply each one by W<sub>E</sub>. Correct, but slow.<br><br>
    The ViT approach: use <code>nn.Conv2d(kernel_size=16, stride=16)</code>.
    Imagine sliding a 16&times;16 stamp across the image with no overlap (stride = 16).
    Each stamp position is exactly one patch, and the Conv2d applies the same linear
    multiplication at every position, all 196 at once in one GPU operation.<br><br>
    The math is <em>identical</em>. The speed difference can be 50&times; or more on a GPU.
    """)

    # ── Zone 2: Thumbnail + demo button ─────────────────────────
    st.markdown("### See it on our image")
    thumb_col, btn_col = st.columns([1, 3])

    class PatchEmbedding(nn.Module):
        def __init__(self, patch_size=16, embed_dim=768):
            super().__init__()
            self.projection = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        def forward(self, x):
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)
            return x

    with thumb_col:
        st.image(img_resized, caption="Input image", use_container_width=True)

    with btn_col:
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
            "margin-bottom:12px;'>Runs the <code>PatchEmbedding</code> layer on our image "
            "and shows what the resulting embedding vectors look like.</p>",
            unsafe_allow_html=True
        )
        if st.button("Run PatchEmbedding forward pass", type="primary"):
            with st.spinner("Running..."):
                embed_model = PatchEmbedding()
                with torch.no_grad():
                    out = embed_model(img_tensor)

                st.success(f"Output shape: {tuple(out.shape)}  ·  196 patches, each a 768-dim vector")

                st.markdown("**Embedding vectors for the first 8 patches (first 64 of 768 dims shown):**")
                st.caption("Each colored line is one patch. Notice how different patches produce very different patterns; the embedding has learned to separate different visual regions.")
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                colors = ['#4f46e5','#059669','#d85a30','#d4537e',
                          '#b45309','#0284c7','#7c3aed','#0f766e']
                for i in range(8):
                    vals = out[0, i, :64].detach().numpy()
                    ax.plot(vals, color=colors[i], alpha=0.85, linewidth=1.3,
                            label=f'Patch {i+1}')
                ax.legend(loc='upper right', fontsize=7,
                          facecolor='white', edgecolor='#e2e8f0',
                          labelcolor='#1e293b', ncol=2)
                ax.set_xlabel("Embedding dimension (0 to 63)", color='#475569', fontsize=9)
                ax.set_ylabel("Value", color='#475569', fontsize=9)
                ax.tick_params(colors='#64748b', labelsize=8)
                ax.spines[:].set_color('#e2e8f0')
                ax.set_title("Each line = one patch's embedding vector", color='#0f172a', fontsize=10)
                st.pyplot(fig)
                plt.close()

                st.markdown("**First 8 learned projection filters (what the network looks for in each patch):**")
                st.markdown(
                    "<p style='font-family:Inter,sans-serif;font-size:0.8rem;color:#64748b;margin:0 0 8px;'>"
                    "These 16&times;16 filters are the first 8 rows of W<sub>E</sub>, reshaped back to image form. "
                    "They start random; after training they detect edges, colors, and textures.</p>",
                    unsafe_allow_html=True
                )
                w = embed_model.projection.weight
                fig2, axes = plt.subplots(2, 4, figsize=(8, 4))
                fig2.patch.set_facecolor('white')
                for idx, ax in enumerate(axes.flatten()):
                    filt = w[idx].detach().numpy()
                    filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
                    filt = filt.transpose(1, 2, 0)
                    ax.imshow(filt)
                    ax.set_title(f'Filter {idx}', color='#0f172a', fontsize=8)
                    ax.axis('off')
                    ax.set_facecolor('white')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
        else:
            info("Click the button above to run the real forward pass on our image.")

    # ── Zone 3: PyTorch implementation ───────────────────────────
    st.divider()
    st.markdown("### PyTorch implementation")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:8px;'>The full <code>PatchEmbedding</code> class. "
        "The single <code>Conv2d</code> line does everything described above:</p>",
        unsafe_allow_html=True
    )
    st.code("""
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2  # 196

        # One Conv2d replaces 196 separate linear operations.
        # kernel_size=patch_size, stride=patch_size means the filter
        # slides exactly patch_size pixels each step, no overlap.
        self.projection = nn.Conv2d(
            in_channels,        # 3  (RGB input)
            embed_dim,          # 768 (output embedding size)
            kernel_size=patch_size,
            stride=patch_size
        )
        # Weight shape: (768, 3, 16, 16) = W_E reshaped for Conv2d

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.projection(x)   # (B, 768, 14, 14): 14x14 grid of embeddings
        x = x.flatten(2)         # (B, 768, 196):    flatten the spatial dims
        x = x.transpose(1, 2)    # (B, 196, 768):    standard sequence-first order
        return x                 # 196 patch embeddings, each 768-dim
    """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CLS TOKEN + POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("4"):
    section_header("4", "CLS Token + Positional Encoding",
                   "Preparing the full 197-token sequence before the transformer encoder")

    # ── Zone 1: Concept ──────────────────────────────────────────
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("### CLS token")
        st.markdown("""
        A **learnable vector** prepended to the patch sequence at position 0.
        After 12 encoder layers, only the CLS token output is passed to the
        classification head. Through self-attention across all 196 patches it
        accumulates a global summary of the image.

        Think of it as a designated note-taker that attends every patch and
        synthesizes the whole scene into a single 768-dim vector.
        """)
    with c_right:
        st.markdown("### Positional encoding")
        st.markdown("""
        Unlike the original Transformer (sinusoidal), ViT uses **learned 1D
        positional embeddings**: one 768-dim vector per position (0 = CLS,
        1-196 = patch positions), added element-wise to the patch embeddings.

        A key finding from the paper: despite being 1D, the model spontaneously
        learns 2D spatial structure in these embeddings after training.
        Patches that are close together in the image end up with similar
        positional embeddings, even though the model was never told their
        2D coordinates.
        """)

    st.divider()

    # ── Zone 2: Demo ─────────────────────────────────────────────
    st.markdown("### Run the embedding pipeline")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:14px;'>Runs our image through patch embedding, prepends the "
        "CLS token, and adds positional encodings. Then visualizes the cosine "
        "similarity structure of the positional embeddings.</p>",
        unsafe_allow_html=True
    )

    thumb_c, btn_c = st.columns([1, 4])
    with thumb_c:
        st.image(user_img.resize((224, 224)), caption="Input", use_container_width=True)

    class _PE4(nn.Module):
        def __init__(self, patch_size=16, embed_dim=768):
            super().__init__()
            self.projection = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        def forward(self, x):
            return self.projection(x).flatten(2).transpose(1, 2)

    class _ViTEmb4(nn.Module):
        def __init__(self, num_patches=196, embed_dim=768):
            super().__init__()
            self.patch_embed = _PE4()
            self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        def forward(self, x):
            B = x.shape[0]
            x   = self.patch_embed(x)
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)
            return x + self.pos_embed

    with btn_c:
        run_emb = st.button("Run ViTEmbedding forward pass", type="primary")

    if run_emb:
        with st.spinner("Running embedding pipeline..."):
            img_tensor = preprocess_image(user_img.resize((224, 224)))
            emb_model  = _ViTEmb4()
            with torch.no_grad():
                out = emb_model(img_tensor)

        st.success(f"Output shape: {tuple(out.shape)}  (197 tokens = 1 CLS + 196 patches, 768 dims each)")

        # ── Positional similarity visualizations ─────────────────
        pos      = emb_model.pos_embed[0].detach()           # (197, 768)
        pos_norm = pos / (pos.norm(dim=-1, keepdim=True) + 1e-8)
        sim      = (pos_norm @ pos_norm.T).numpy()           # (197, 197)

        st.markdown("### Positional embedding cosine similarity")
        st.markdown("""
        <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:6px;'>
        Cosine similarity measures how similar two vectors are: +1 means identical direction,
        0 means orthogonal, -1 means opposite. We compute this between every pair of the 197
        positional embedding vectors and display two views below.
        </p>
        <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#6366f1;margin-bottom:14px;'>
        <strong>Note:</strong> The right plot uses <em>randomly initialized</em> embeddings here,
        so it shows near-zero, noisy values. In a pretrained ViT it becomes a smooth spatial
        gradient, which is the key insight from the paper.
        </p>
        """, unsafe_allow_html=True)

        # Center patch is at grid (row=6, col=6) = patch index 90 = pos_embed index 91
        center_idx = 1 + 6 * 14 + 6   # = 91
        center     = sim[center_idx, 1:].reshape(14, 14)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('white')

        # Left: 1D position index map arranged as 14x14 spatial grid
        pos_indices = np.arange(196).reshape(14, 14)
        im1 = axes[0].imshow(pos_indices, cmap='plasma',
                             interpolation='nearest', aspect='equal')
        axes[0].set_title("The 1D position index each patch receives\n"
                          "(0 = top-left, 195 = bottom-right)",
                          color='#0f172a', fontsize=10, pad=8)
        axes[0].set_xlabel("Patch column (0-13)", color='#475569', fontsize=9)
        axes[0].set_ylabel("Patch row (0-13)", color='#475569', fontsize=9)
        axes[0].set_xticks(range(0, 14, 2))
        axes[0].set_yticks(range(0, 14, 2))
        axes[0].tick_params(colors='#64748b', labelsize=8)
        axes[0].set_facecolor('white')
        cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=8, colors='#64748b')
        cb1.set_label("Position index (0-195)", fontsize=8, color='#475569')
        # Annotate a few corners
        for (r, c, lbl) in [(0,0,'0'), (0,13,'13'), (13,0,'182'), (13,13,'195')]:
            axes[0].text(c, r, lbl, ha='center', va='center',
                         fontsize=6, color='white', fontweight='bold')

        # Right: center patch similarity to all patches arranged in 14x14 grid
        im2 = axes[1].imshow(center, cmap='RdYlBu_r', vmin=-1, vmax=1,
                             interpolation='nearest', aspect='equal')
        axes[1].set_title("Center patch (row 6, col 6) cosine similarity\n"
                          "to every other patch position in the 14x14 grid",
                          color='#0f172a', fontsize=10, pad=8)
        axes[1].set_xlabel("Patch column (0-13)", color='#475569', fontsize=9)
        axes[1].set_ylabel("Patch row (0-13)", color='#475569', fontsize=9)
        axes[1].set_xticks(range(14)); axes[1].set_yticks(range(14))
        axes[1].tick_params(colors='#64748b', labelsize=7)
        axes[1].set_facecolor('white')
        # Star on center patch: black outline, yellow fill so it's always visible
        axes[1].plot(6, 6, marker='*', color='#fbbf24', markersize=18,
                     markeredgecolor='#1e1b4b', markeredgewidth=1.5, zorder=5)
        cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=8, colors='#64748b')
        cb2.set_label("Cosine similarity", fontsize=8, color='#475569')

        plt.tight_layout(pad=2.0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        <div class="enc-cards" style="margin-top:16px;">
          <div class="enc-card" style="background:#eef2ff;border-left:4px solid #4f46e5;">
            <div class="enc-card-title" style="color:#3730a3;">Left plot: what the model starts with</div>
            <div class="enc-card-body">
              ViT assigns each patch a 1D integer index (0 = top-left, 195 = bottom-right).
              This plot maps those indices back onto the 2D grid so you can see which
              number corresponds to which spatial location. The model receives these
              numbers as raw inputs; it has no built-in knowledge that position 14 is
              directly below position 0. It must learn that from data.
            </div>
          </div>
          <div class="enc-card" style="background:#f0fdf4;border-left:4px solid #16a34a;">
            <div class="enc-card-title" style="color:#14532d;">Right plot: what the model learns (with random weights here)</div>
            <div class="enc-card-body">
              The gold star marks the center patch (row 6, col 6). Each cell shows the
              cosine similarity of the center patch's positional embedding to every other
              patch's positional embedding, arranged spatially. With random initialization
              (shown here) the values are near zero with noise. In a
              <strong>pretrained ViT</strong> this becomes a smooth gradient: warm colors
              near the star, cool at the corners, proving the model learned that nearby
              patches should have similar positional encodings even though it only received
              a flat 1D index as input.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        info("Click the button to run the full embedding pipeline on our image.")

    # ── Zone 3: Code (collapsed) ─────────────────────────────────
    with st.expander("Show the ViTEmbedding implementation", expanded=False):
        st.code("""
class ViTEmbedding(nn.Module):
    def __init__(self, num_patches=196, embed_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbedding()   # from Section 3

        # Learnable CLS token: shape (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learned positional embeddings: one vector per position (197 total)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)

    def forward(self, x):
        B = x.shape[0]

        x   = self.patch_embed(x)              # (B, 196, 768)
        cls = self.cls_token.expand(B, -1, -1) # (B,   1, 768)
        x   = torch.cat([cls, x], dim=1)       # (B, 197, 768)
        x   = x + self.pos_embed               # (B, 197, 768) -- ready for encoder
        return x
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SELF-ATTENTION
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("5"):
    section_header("5", "Self-Attention over Image Patches",
                   "How every patch looks at every other patch, and what that reveals")

    # ── Zone 1: What is self-attention? ──────────────────────────
    col_concept, col_qkv = st.columns([1, 1])

    with col_concept:
        st.markdown("### What is self-attention?")
        st.markdown("""
        In a CNN, each neuron only sees a small local neighborhood.
        **Self-attention is different:** every patch looks at every other patch
        in the image simultaneously and decides how much to "pay attention" to each one.

        After one attention layer, each patch has gathered information from the
        entire image, weighted by relevance. This is how ViT can relate a patch
        of sky to a wing tip 200 pixels away, something CNNs cannot do in one step.

        With **12 encoder layers**, each patch refines its representation 12 times,
        each time re-reading the whole image through a different lens.
        """)
        info("""
        <strong>Note:</strong> the output of self-attention for patch <em>i</em>
        is a weighted average of all patch Values, where the weights come from how
        well patch <em>i</em>'s Query matches every other patch's Key.
        Patches with high match scores contribute more to the result.
        """)

    with col_qkv:
        st.markdown("### Queries, Keys, and Values")
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.855rem;color:#374151;"
            "margin-bottom:10px;'>Every patch is projected into three vectors. "
            "Think of them like a search engine:</p>",
            unsafe_allow_html=True
        )
        st.markdown("""
        <div class="qkv-grid">
          <div class="qkv-card q-card">
            <div class="qkv-title">Query (Q)</div>
            <div class="qkv-body"><span class="qkv-em">"What am I looking for?"</span><br><br>
            Each patch broadcasts what kind of information it needs from the rest of the image.</div>
          </div>
          <div class="qkv-card k-card">
            <div class="qkv-title">Key (K)</div>
            <div class="qkv-body"><span class="qkv-em">"What do I contain?"</span><br><br>
            Each patch announces its content so other patches can decide whether to attend to it.</div>
          </div>
          <div class="qkv-card v-card">
            <div class="qkv-title">Value (V)</div>
            <div class="qkv-body"><span class="qkv-em">"Here is my information."</span><br><br>
            The actual content that gets transferred when a Query strongly matches a Key.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **The math in one line:**
        Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V

        The dot product Q · Kᵀ measures similarity between every pair of patches.
        Dividing by √d (= √64 = 8) prevents the values from getting too large before softmax.
        Softmax turns scores into probabilities that sum to 1.
        """)

    st.divider()

    # ── Zone 2: Multi-head + what pretrained means ───────────────
    col_mh, col_pt = st.columns([1, 1])

    with col_mh:
        st.markdown("### Why 12 heads?")
        st.markdown("""
        Running attention once gives one "view" of the image.
        **Multi-head attention** runs 12 independent attention operations in parallel,
        each using its own learned Q, K, V weight matrices.

        Each head specializes in something different:
        - One head may focus on **local edges and textures**
        - Another on **object parts** (wing to body, eye to face)
        - Another on **background vs foreground** separation
        - Others on **color regions**, **symmetry**, or **semantic categories**

        The 12 outputs are concatenated and projected back to 768 dims.
        The model learns which heads to trust for which decisions.
        """)

    with col_pt:
        st.markdown("### What does &ldquo;pretrained&rdquo; mean?")
        st.markdown("""
        <div class="pretrained-card">
          <h4>Loading google/vit-base-patch16-224-in21k</h4>
          <div class="pretrained-row">
            <div class="pretrained-icon">&#128190;</div>
            <div class="pretrained-text">
              <strong>What we download (~330 MB):</strong> the full set of learned weights
              for all 12 encoder layers, the patch embedding, and the CLS token.
              86 million numbers saved to disk.
            </div>
          </div>
          <div class="pretrained-row">
            <div class="pretrained-icon">&#127970;</div>
            <div class="pretrained-text">
              <strong>How it was trained:</strong> on ImageNet-21k, 14 million images
              across 21,841 categories. Training ran for weeks on Google TPU clusters.
            </div>
          </div>
          <div class="pretrained-row">
            <div class="pretrained-icon">&#128161;</div>
            <div class="pretrained-text">
              <strong>Why we use it:</strong> those weights already encode how to detect
              edges, textures, shapes, and objects. We skip the weeks of training
              and go straight to reading the learned attention patterns.
            </div>
          </div>
          <div class="pretrained-row">
            <div class="pretrained-icon">&#128269;</div>
            <div class="pretrained-text">
              <strong>What we do with it:</strong> pass our image through all 12 layers
              and capture the 197&times;197 attention matrix at every layer and every head.
              We then visualize which patches the model focused on.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    warn("""
    <strong>First run only:</strong> the pretrained model (~330 MB) will be downloaded
    and cached locally. Every run after that loads instantly from cache.
    On CPU, inference takes roughly 20-30 seconds.
    """)

    # ── Zone 3: Demo ─────────────────────────────────────────────
    st.markdown("### Run it on our image")

    img_resized = user_img.resize((224, 224))

    thumb_col, ctrl_col = st.columns([1, 3])

    with thumb_col:
        st.image(img_resized, caption="Our image", use_container_width=True)

    with ctrl_col:
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.855rem;color:#374151;"
            "margin-bottom:12px;'>"
            "Choose a layer and head, then click the button. "
            "Layer 0 captures low-level features (edges, colors). "
            "Layer 11 captures high-level semantics (objects, categories). "
            "Different heads specialize in different relationships.</p>",
            unsafe_allow_html=True
        )
        layer_idx = st.slider("Encoder layer (0 = first, 11 = last)", 0, 11, 11)
        head_idx  = st.slider("Attention head (0-11)", 0, 11, 0)

        if st.button("Load pretrained ViT and extract attention", type="primary"):
            with st.spinner("Loading pretrained ViT-Base/16 (first run downloads ~330 MB)..."):
                try:
                    model, processor = load_vit_model()
                    inputs = processor(images=img_resized, return_tensors="pt")

                    with torch.no_grad():
                        outputs = model(**inputs, output_attentions=True)

                    attentions = outputs.attentions
                    # attentions: tuple of 12 tensors, each (1, 12, 197, 197)

                    st.success(
                        f"Model loaded. Showing layer {layer_idx}, head {head_idx}. "
                        f"Full attention tensor per layer: {tuple(attentions[0].shape)}"
                    )

                    attn = attentions[layer_idx][0, head_idx].detach().numpy()

                    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
                    fig.patch.set_facecolor('white')

                    # Panel 1: raw attention matrix (first 20 tokens)
                    im1 = axes[0].imshow(attn[:20, :20], cmap='magma', vmin=0)
                    axes[0].set_title(
                        f"Attention matrix (first 20 tokens)\nlayer {layer_idx}, head {head_idx}",
                        color='#0f172a', fontsize=9)
                    axes[0].set_xlabel("Attends TO", color='#64748b', fontsize=8)
                    axes[0].set_ylabel("Attends FROM", color='#64748b', fontsize=8)
                    axes[0].tick_params(colors='#64748b', labelsize=7)
                    axes[0].set_facecolor('white')
                    plt.colorbar(im1, ax=axes[0])

                    # Panel 2: CLS token attention over 14x14 patch grid
                    cls_attn = attn[0, 1:]        # row 0 = CLS, skip itself
                    cls_grid = cls_attn.reshape(14, 14)
                    im2 = axes[1].imshow(cls_grid, cmap='viridis')
                    axes[1].set_title(
                        "CLS attention over 14x14 patch grid\n(brighter = more attended)",
                        color='#0f172a', fontsize=9)
                    axes[1].tick_params(colors='#64748b', labelsize=7)
                    axes[1].set_facecolor('white')
                    plt.colorbar(im2, ax=axes[1])

                    # Panel 3: overlay on original image
                    img_arr = np.array(img_resized)
                    import cv2
                    attn_map  = cv2.resize(cls_grid, (224, 224))
                    attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                    axes[2].imshow(img_arr)
                    axes[2].imshow(attn_norm, cmap='hot', alpha=0.5)
                    axes[2].set_title(
                        "CLS attention overlaid on image\n(bright red = high attention)",
                        color='#0f172a', fontsize=9)
                    axes[2].axis('off')
                    axes[2].set_facecolor('white')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    st.markdown(
                        "<p style='font-family:Inter,sans-serif;font-size:0.855rem;"
                        "color:#374151;margin:8px 0;'>"
                        "The left panel shows the raw 197&times;197 attention matrix. "
                        "Row <em>i</em>, column <em>j</em> = how much token <em>i</em> attends to token <em>j</em>. "
                        "The middle and right panels show only the CLS token's row (row 0): "
                        "which of the 196 patches the model considered most important for classification.</p>",
                        unsafe_allow_html=True
                    )

                    # Attention rollout
                    st.markdown("### Attention rollout: the full picture across all 12 layers")
                    st.markdown(
                        "<p style='font-family:Inter,sans-serif;font-size:0.855rem;color:#374151;"
                        "margin-bottom:10px;'>"
                        "A single layer's attention map shows only what happened at that layer. "
                        "But information flows through all 12 layers, each re-mixing the previous output. "
                        "<strong>Attention rollout</strong> multiplies the attention matrices from all 12 layers "
                        "together (accounting for residual connections), giving a single map that shows "
                        "which input patches had the most cumulative influence on the CLS token's "
                        "final representation. This is the closest we can get to asking: "
                        "<em>what did the model actually look at?</em></p>",
                        unsafe_allow_html=True
                    )

                    rollout = torch.eye(197)
                    for layer_attn in attentions:
                        a = layer_attn[0].mean(0).detach()   # average over 12 heads
                        a = a + torch.eye(197)                # add residual connection
                        a = a / a.sum(dim=-1, keepdim=True)   # re-normalize rows
                        rollout = a @ rollout

                    rollout_cls  = rollout[0, 1:].numpy().reshape(14, 14)
                    rollout_norm = (rollout_cls - rollout_cls.min()) / (rollout_cls.max() - rollout_cls.min() + 1e-8)
                    rollout_up   = cv2.resize(rollout_norm, (224, 224))

                    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
                    fig2.patch.set_facecolor('white')

                    axes2[0].imshow(rollout_cls, cmap='inferno')
                    axes2[0].set_title("Rollout heatmap (14x14 patch grid)", color='#0f172a', fontsize=10)
                    axes2[0].tick_params(colors='#64748b', labelsize=8)
                    axes2[0].set_facecolor('white')

                    axes2[1].imshow(img_arr)
                    axes2[1].imshow(rollout_up, cmap='inferno', alpha=0.55)
                    axes2[1].set_title("Rollout overlay: what ViT sees", color='#0f172a', fontsize=10)
                    axes2[1].axis('off')

                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

                    success("""
                    Bright regions in the rollout map are the patches that most influenced
                    the final CLS output; that is the model's focus area for this image.
                    Unlike a CNN's fixed receptive field, ViT's focus is entirely data-driven
                    and can jump anywhere in the image.
                    """)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Make sure you have: pip install transformers torch torchvision opencv-python-headless")
        else:
            info("Click the button above to load the pretrained model and visualize real attention maps on our image.")

    # ── Zone 4: Code reference (collapsed) ──────────────────────
    with st.expander("Show the attention computation code", expanded=False):
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
            "margin-bottom:8px;'>Scaled dot-product attention at the heart of every "
            "encoder layer. Each of the 12 heads runs this independently with its own weights:</p>",
            unsafe_allow_html=True
        )
        st.code("""
# Input: x shape (B, 197, 768)   -- 197 = 196 patches + 1 CLS token
B, N, D = x.shape   # B=batch, N=197, D=768

# Step 1: project x into Q, K, V with a single fused linear layer
qkv = self.qkv(x)                                  # (B, 197, 3*768)
qkv = qkv.reshape(B, N, 3, num_heads, D//num_heads)
qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, heads, N, 64)
q, k, v = qkv.unbind(0)                            # each: (B, 12, 197, 64)

# Step 2: scaled dot-product attention
scale = (D // num_heads) ** -0.5   # 1/sqrt(64) = 0.125
attn  = (q @ k.transpose(-2, -1)) * scale   # (B, 12, 197, 197)
attn  = attn.softmax(dim=-1)                # each row sums to 1

# Step 3: weighted sum of Values
out = (attn @ v)                    # (B, 12, 197, 64)
out = out.transpose(1, 2)           # (B, 197, 12, 64)
out = out.reshape(B, N, D)          # (B, 197, 768)
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ENCODER BLOCK
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("6"):
    section_header("6", "Transformer Encoder Block",
                   "The repeating unit stacked 12 times to build ViT-Base")

    # ── Zone 1: The big picture ──────────────────────────────────
    st.markdown("### What does an encoder block do?")
    col_intro, col_eq = st.columns([1.2, 1])

    with col_intro:
        st.markdown("""
        The encoder block is ViT's core processing unit. It is stacked **12 times** in sequence.
        Each block receives the full sequence of 197 tokens (196 patches + CLS),
        processes it, and passes it along with the **same shape in and out**.

        Every block does two things in order:
        1. **Multi-head self-attention**: every token reads the whole sequence and updates itself
        2. **Feed-forward network (FFN)**: each token is processed independently through a small MLP

        Each of those two operations is wrapped in a **residual connection** and preceded by a **LayerNorm**.
        That is the entire block. The elegance is in the repetition: 12 identical blocks,
        each refining the representations a little more.
        """)

    with col_eq:
        st.markdown("### The whole block in two lines")
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.855rem;color:#374151;"
            "margin-bottom:8px;'>The entire <code>TransformerBlock.forward()</code> "
            "reduces to just two equations:</p>",
            unsafe_allow_html=True
        )
        st.code("""
x = x + Attention( LayerNorm(x) )   # sublayer 1
x = x + FFN(       LayerNorm(x) )   # sublayer 2
        """, language="python")
        st.markdown(
            "<p style='font-family:Inter,sans-serif;font-size:0.835rem;color:#374151;"
            "margin-top:6px;'>"
            "The <code>x +</code> at the start of each line is the <strong>residual connection</strong>. "
            "<code>LayerNorm</code> comes first (pre-norm). "
            "Shape <code>(B, 197, 768)</code> enters and exits unchanged.</p>",
            unsafe_allow_html=True
        )

    st.divider()

    # ── Zone 2: Four component cards ────────────────────────────
    st.markdown("### Inside each block: four components")
    st.markdown("""
    <div class="enc-cards">

      <div class="enc-card" style="background:#eef2ff;border-left:4px solid #4f46e5;">
        <div class="enc-card-title" style="color:#3730a3;">Residual Connection (+)</div>
        <div class="enc-card-body">
          Adds the block's input directly to its output: <code>x = x + sublayer(x)</code>.<br><br>
          This creates a shortcut that lets gradients flow straight back through 12 layers
          without vanishing. It also means each block only needs to learn a small
          <em>correction</em> to its input, not a completely new representation.
          Deep networks are practically untrainable without this.
        </div>
      </div>

      <div class="enc-card" style="background:#dcfce7;border-left:4px solid #16a34a;">
        <div class="enc-card-title" style="color:#14532d;">Layer Normalization</div>
        <div class="enc-card-body">
          Re-scales each token's 768-dim vector so it has mean 0 and std 1,
          then applies learned scale &amp; shift parameters.<br><br>
          Applied <em>before</em> each sublayer (pre-norm). This stabilizes training
          by preventing individual dimensions from growing too large.
          Think of it as a reset button before each major operation.
        </div>
      </div>

      <div class="enc-card" style="background:#fffbeb;border-left:4px solid #d97706;">
        <div class="enc-card-title" style="color:#92400e;">Feed-Forward Network (FFN)</div>
        <div class="enc-card-body">
          Two linear layers with GELU between them, applied to <em>each token independently</em>:
          <span class="enc-card-formula">768 &rarr; 3072 &rarr; 768  (ratio = 4&times;)</span>
          The hidden dim of 3072 = 4&times;768 is a hyperparameter called <code>mlp_ratio</code>.
          While attention mixes information <em>across</em> tokens, FFN processes
          each token <em>within itself</em>. Together they cover both dimensions.
        </div>
      </div>

      <div class="enc-card" style="background:#fdf4ff;border-left:4px solid #a855f7;">
        <div class="enc-card-title" style="color:#7e22ce;">GELU Activation</div>
        <div class="enc-card-body">
          Gaussian Error Linear Unit: a smooth, differentiable activation that
          approximates ReLU but passes small negative values through with a slight
          downweighting rather than zeroing them.<br><br>
          ViT uses GELU everywhere instead of ReLU because the smooth curve
          produces better gradients and slightly better accuracy at scale.
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    # ── Zone 3: Pre-norm vs post-norm diagram ───────────────────
    st.markdown("### Pre-norm vs post-norm: where LayerNorm goes")
    st.markdown("""
    <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:12px;'>
    The original transformer (Vaswani et al. 2017) placed LayerNorm <em>after</em> the residual add
    (post-norm). ViT places it <em>before</em> the sublayer (pre-norm).
    Pre-norm produces more stable gradients at large scale and is the standard in modern transformers.
    </p>
    <div class="norm-compare">

      <div class="norm-col">
        <div class="norm-header post">Post-norm &nbsp; (original Transformer, 2017)</div>
        <div class="norm-row"><div class="nb nb-input">Input x</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-attn">Multi-Head Attention</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-resid">+ Residual add</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-ln-post">LayerNorm &nbsp;&#9650; applied here</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-ffn">Feed-Forward Network</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-resid">+ Residual add</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-ln-post">LayerNorm &nbsp;&#9650; applied here</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-output">Output</div></div>
        <div class="norm-note postnorm-note">
          Norm sees a mix of raw residual + sublayer output.<br>
          Can destabilize at large scale.
        </div>
      </div>

      <div class="norm-col">
        <div class="norm-header pre">Pre-norm &nbsp; (ViT &amp; modern transformers)</div>
        <div class="norm-row"><div class="nb nb-input">Input x</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-ln-pre">LayerNorm &nbsp;&#9660; applied first</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-attn">Multi-Head Attention</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-resid">+ Residual add</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-ln-pre">LayerNorm &nbsp;&#9660; applied first</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-ffn">Feed-Forward Network</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-resid">+ Residual add</div></div>
        <div class="norm-arrow">&#8595;</div>
        <div class="norm-row"><div class="nb nb-output">Output</div></div>
        <div class="norm-note">
          Norm always sees a clean, bounded input.<br>
          Stable gradients; preferred at scale.
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    info("""
    <strong>Why does the order matter?</strong>
    In post-norm, LayerNorm operates on the sum of the residual and the sublayer output.
    Early in training those sums can be very large, causing unstable gradient magnitudes.
    Pre-norm normalizes the input <em>before</em> the sublayer sees it, so activations
    stay bounded regardless of what the previous layer produced.
    At ViT-Base scale (12 layers) the difference is modest; at larger models it is critical.
    """)

    # ── Zone 4: Demo ─────────────────────────────────────────────
    st.divider()
    st.markdown("### Run a TransformerBlock")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:14px;'>The demo uses a randomly initialized block on a random "
        "(1, 197, 768) tensor, the same shape the real model sees after Section 4. "
        "The key result to watch: <strong>shape is perfectly preserved</strong>, "
        "and each block has about 7 million parameters.</p>",
        unsafe_allow_html=True
    )

    if st.button("Run TransformerBlock forward pass", type="primary"):
        with st.spinner("Running..."):
            class _MHA(nn.Module):
                def __init__(self, d=768, h=12):
                    super().__init__()
                    self.h = h; self.scale = (d//h)**-0.5
                    self.qkv  = nn.Linear(d, d*3, bias=False)
                    self.proj = nn.Linear(d, d)
                def forward(self, x):
                    B, N, D = x.shape; H = self.h
                    qkv = self.qkv(x).reshape(B,N,3,H,D//H).permute(2,0,3,1,4)
                    q, k, v = qkv.unbind(0)
                    attn = (q @ k.transpose(-2,-1)) * self.scale
                    attn = attn.softmax(-1)
                    return self.proj((attn @ v).transpose(1,2).reshape(B,N,D))

            class _Block(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(768)
                    self.attn  = _MHA()
                    self.norm2 = nn.LayerNorm(768)
                    self.mlp   = nn.Sequential(
                        nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768)
                    )
                def forward(self, x):
                    x = x + self.attn(self.norm1(x))
                    x = x + self.mlp(self.norm2(x))
                    return x

            x     = torch.randn(1, 197, 768)
            block = _Block()
            with torch.no_grad():
                out = block(x)

            c1, c2 = st.columns(2)
            c1.success(f"Input shape:  {tuple(x.shape)}")
            c2.success(f"Output shape: {tuple(out.shape)}  (unchanged)")

            st.code(f"""
Input   | mean: {x.mean().item():.4f}   std: {x.std().item():.4f}
Output  | mean: {out.mean().item():.4f}   std: {out.std().item():.4f}
            """)

            params      = sum(p.numel() for p in block.parameters())
            attn_params = sum(p.numel() for p in block.attn.parameters())
            mlp_params  = sum(p.numel() for p in block.mlp.parameters())
            norm_params = sum(p.numel() for p in block.norm1.parameters()) * 2

            st.markdown(f"""
            | Sub-module | Parameters |
            |---|---|
            | Multi-Head Attention | {attn_params:,} |
            | Feed-Forward Network | {mlp_params:,} |
            | LayerNorm (×2) | {norm_params:,} |
            | **One block total** | **{params:,}** |
            | **All 12 blocks** | **{params*12:,}** |
            """)
    else:
        info("Click the button to run one TransformerBlock forward pass on a random (1, 197, 768) tensor.")

    # ── Zone 5: Code (collapsed) ─────────────────────────────────
    with st.expander("Show the full TransformerBlock implementation", expanded=False):
        st.code("""
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (embed_dim // num_heads) ** -0.5   # 1/sqrt(64)
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, H, D//H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                     # each (B, 12, 197, 64)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)                  # (B, 12, 197, 197)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden = int(embed_dim * mlp_ratio)      # 3072

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # pre-norm attention sublayer
        x = x + self.mlp(self.norm2(x))    # pre-norm FFN sublayer
        return x                            # shape unchanged: (B, 197, 768)
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MLP HEAD
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("7"):
    section_header("7", "MLP Classification Head",
                   "Converting the CLS token into a class prediction")

    # ── Zone 1: What is the head? ────────────────────────────────
    st.markdown("### Where we are in the pipeline")
    st.markdown("""
    <div class="pipeline">
      <div class="pipe-step">
        <div class="pipe-box">(B, 197, 768)<small>encoder output<br>197 tokens</small></div>
        <div class="pipe-label">After 12 blocks</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-step">
        <div class="pipe-box">x[:, 0]<small>take position 0<br>CLS token only</small></div>
        <div class="pipe-label">Extract CLS</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-step">
        <div class="pipe-box">(B, 768)<small>one vector per<br>image in batch</small></div>
        <div class="pipe-label">Image summary</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-step">
        <div class="pipe-box">Linear head<small>learned weights<br>768 &rarr; num_classes</small></div>
        <div class="pipe-label">Classification</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-step">
        <div class="pipe-box">(B, num_classes)<small>raw scores<br>one per class</small></div>
        <div class="pipe-label">Logits</div>
      </div>
      <div class="pipe-arrow">&#8594;</div>
      <div class="pipe-step">
        <div class="pipe-box">Softmax<small>convert to<br>probabilities</small></div>
        <div class="pipe-label">Prediction</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_what, col_cls = st.columns([1, 1])
    with col_what:
        st.markdown("### Why only the CLS token?")
        st.markdown("""
        After 12 encoder blocks, we have 197 output vectors of 768 dims each.
        We only need **one** vector to classify the image.

        The CLS token was designed exactly for this: placed at position 0 before
        the encoder, it has no patch to represent. Its only job is to
        **attend to every patch across all 12 layers** and accumulate a
        global summary of the image.

        By the final layer, `x[:, 0]` is a rich 768-dim fingerprint of the
        entire image. The other 196 patch vectors are discarded at this stage.
        """)
    with col_cls:
        st.markdown("### What are logits and softmax?")
        st.markdown("""
        The linear head multiplies the 768-dim CLS vector by a weight matrix
        to produce **N raw scores**, one per class. These are called **logits**.

        Logits are unbounded numbers; a higher logit means the model is more
        confident about that class, but the values have no direct interpretation
        as probabilities.

        **Softmax** converts logits to probabilities:
        - All values become positive (via exponential)
        - All values sum to exactly 1.0
        - The class with the highest logit gets the highest probability

        The final **predicted class** is the `argmax` of the probabilities.
        """)

    st.divider()

    # ── Zone 2: Two head modes ───────────────────────────────────
    st.markdown("### Pre-training head vs fine-tuning head")
    st.markdown("""
    <div class="enc-cards">

      <div class="enc-card" style="background:#f0f9ff;border-left:4px solid #0284c7;">
        <div class="enc-card-title" style="color:#0c4a6e;">Pre-training head &nbsp; (training from scratch)</div>
        <div class="enc-card-body">
          Used when training on a large dataset like ImageNet-21k (21,841 classes).<br><br>
          Architecture: a <strong>2-layer MLP</strong><br>
          <span class="enc-card-formula">768 &rarr; 3072 &rarr; GELU &rarr; num_classes</span>
          The extra hidden layer gives the model more capacity to map the CLS
          representation to the right class during initial training.
          After pre-training, this head is <em>discarded</em> when transferring to a new task.
        </div>
      </div>

      <div class="enc-card" style="background:#f0fdf4;border-left:4px solid #16a34a;">
        <div class="enc-card-title" style="color:#14532d;">Fine-tuning head &nbsp; (transfer learning)</div>
        <div class="enc-card-body">
          Used when adapting the pretrained ViT to a new downstream task
          (e.g., medical imaging with 10 classes, satellite imagery, etc.).<br><br>
          Architecture: a <strong>single linear layer</strong><br>
          <span class="enc-card-formula">768 &rarr; num_classes</span>
          The encoder already learned rich general-purpose features.
          A simple linear probe on top is sufficient, and much faster to train.
          This is the standard approach for transfer learning with ViT.
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    info("""
    <strong>Transfer learning in practice:</strong>
    The typical workflow is: (1) download the pretrained encoder weights,
    (2) attach a new single-linear head sized to your task's class count,
    (3) fine-tune the whole model on your data for a few epochs.
    The encoder's 85+ million parameters already encode powerful visual features;
    only the head's ~768 &times; num_classes parameters start from random.
    """)

    st.divider()

    # ── Zone 3: Demo ─────────────────────────────────────────────
    st.markdown("### See it in action")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:14px;'>We feed a random (1, 197, 768) encoder output through the head "
        "and visualize the resulting probabilities. With random weights, the distribution "
        "is nearly uniform; a trained model would show a sharp peak at the correct class.</p>",
        unsafe_allow_html=True
    )

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        num_classes = st.selectbox("Number of classes", [10, 100, 1000], index=0)
    with ctrl2:
        finetune = st.checkbox("Fine-tuning mode (single linear layer)", value=True)

    if st.button("Run MLPHead forward pass", type="primary"):
        with st.spinner("Running..."):

            class _MLPHead(nn.Module):
                def __init__(self, embed_dim=768, num_classes=10,
                             hidden_dim=3072, finetune=True):
                    super().__init__()
                    self.head = (
                        nn.Linear(embed_dim, num_classes)
                        if finetune else
                        nn.Sequential(
                            nn.Linear(embed_dim, hidden_dim),
                            nn.GELU(),
                            nn.Linear(hidden_dim, num_classes)
                        )
                    )
                def forward(self, x):
                    return self.head(x[:, 0])   # extract CLS token first

            encoder_out = torch.randn(1, 197, 768)
            head        = _MLPHead(num_classes=num_classes, finetune=finetune)

            with torch.no_grad():
                logits = head(encoder_out)
                probs  = torch.softmax(logits, dim=-1)

            # Shape walkthrough
            sa, sb, sc = st.columns(3)
            sa.success(f"Encoder output: {tuple(encoder_out.shape)}")
            sb.success(f"CLS extracted:  (1, 768)")
            sc.success(f"Logits / probs: (1, {num_classes})")

            # Logits vs probs side-by-side code block
            top_n = min(8, num_classes)
            top_probs, top_idx = probs[0].topk(top_n)
            top_logits = logits[0][top_idx]
            st.markdown("**Top classes: raw logits vs softmax probabilities**")
            rows = "\n".join(
                f"Class {top_idx[i].item():>4}  |  logit: {top_logits[i].item():+.4f}  |  prob: {top_probs[i].item():.4f}  ({top_probs[i].item()*100:.2f}%)"
                for i in range(top_n)
            )
            st.code(rows)

            # Bar chart: gradient of colors by rank
            palette = ['#4f46e5','#6366f1','#818cf8','#a5b4fc',
                       '#c7d2fe','#dde4ff','#eef0ff','#f5f6ff']
            bar_colors = palette[:top_n]

            fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
            fig.patch.set_facecolor('white')

            labels = [f"Class {i.item()}" for i in top_idx]

            # Left: logits
            axes[0].bar(labels, top_logits.numpy(), color=bar_colors,
                        edgecolor='#e2e8f0', linewidth=0.6)
            axes[0].set_title("Raw logits (before softmax)",
                              color='#0f172a', fontsize=10)
            axes[0].set_ylabel("Logit value", color='#475569', fontsize=9)
            axes[0].axhline(0, color='#94a3b8', linewidth=0.8, linestyle='--')
            axes[0].tick_params(colors='#64748b', labelsize=8,
                                axis='x', rotation=30)
            axes[0].tick_params(colors='#64748b', labelsize=8, axis='y')
            axes[0].spines[:].set_color('#e2e8f0')
            axes[0].set_facecolor('white')

            # Right: probabilities
            bars = axes[1].bar(labels, top_probs.numpy(), color=bar_colors,
                               edgecolor='#e2e8f0', linewidth=0.6)
            for bar, val in zip(bars, top_probs.numpy()):
                axes[1].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.001,
                             f"{val:.3f}", ha='center', va='bottom',
                             fontsize=7.5, color='#374151')
            axes[1].set_title("After softmax (probabilities sum to 1.0)",
                              color='#0f172a', fontsize=10)
            axes[1].set_ylabel("Probability", color='#475569', fontsize=9)
            axes[1].tick_params(colors='#64748b', labelsize=8,
                                axis='x', rotation=30)
            axes[1].tick_params(colors='#64748b', labelsize=8, axis='y')
            axes[1].spines[:].set_color('#e2e8f0')
            axes[1].set_facecolor('white')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            params       = sum(p.numel() for p in head.parameters())
            enc_params   = 86_415_592 - params
            mode_label   = "fine-tuning (linear)" if finetune else "pre-training (MLP)"
            st.markdown(f"""
            | | Parameters |
            |---|---|
            | Head ({mode_label}) | {params:,} |
            | Encoder (all 12 blocks + embeddings) | {enc_params:,} |
            | **Total ViT-Base** | **{86_415_592:,}** |
            """)
            st.caption("The head is tiny: less than 1% of total parameters. The encoder does the heavy lifting.")
    else:
        info("Click the button above to run the head on a random encoder output and visualize logits vs probabilities.")

    # ── Zone 4: Code (collapsed) ─────────────────────────────────
    with st.expander("Show the MLPHead implementation", expanded=False):
        st.code("""
class MLPHead(nn.Module):
    def __init__(self, embed_dim=768, num_classes=1000,
                 hidden_dim=3072, finetune=False):
        super().__init__()
        if finetune:
            # Transfer learning: single linear layer
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            # Pre-training: 2-layer MLP
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),   # 768 -> 3072
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes)  # 3072 -> num_classes
            )

    def forward(self, x):
        # x: (B, 197, 768): full encoder output
        cls_out = x[:, 0]            # (B, 768): CLS token at position 0
        logits  = self.head(cls_out) # (B, num_classes): raw class scores
        return logits
        # Note: softmax is applied outside (in loss function or at inference)
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("8"):
    section_header("8", "Full ViT Model", "Assembling all components: forward pass end to end")

    # ── Zone 1: Capstone pipeline ────────────────────────────────
    st.markdown("### The complete ViT pipeline")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:12px;'>Every section in this walkthrough built one piece. "
        "Here they all connect. Follow the tensor shape through each stage:</p>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-family:Inter,sans-serif;font-size:0.80rem;">
      <thead>
        <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
          <th style="padding:9px 12px;color:#374151;font-weight:700;text-align:center;">Stage</th>
          <th style="padding:9px 12px;color:#374151;font-weight:700;text-align:center;">Operation</th>
          <th style="padding:9px 12px;color:#374151;font-weight:700;text-align:center;">Shape in</th>
          <th style="padding:9px 12px;color:#374151;font-weight:700;text-align:center;">Shape out</th>
          <th style="padding:9px 12px;color:#374151;font-weight:700;text-align:left;">What happens</th>
          <th style="padding:9px 12px;color:#374151;font-weight:700;text-align:center;">Covered in</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:9px 12px;text-align:center;font-weight:700;color:#4f46e5;">1</td>
          <td style="padding:9px 12px;font-weight:600;color:#1e293b;">Patch extraction</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, 3, 224, 224)</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, 196, 768)</td>
          <td style="padding:9px 12px;color:#374151;">Conv2d splits image into 196 non-overlapping 16&times;16 patches; each flattened to 768 dims</td>
          <td style="padding:9px 12px;text-align:center;"><span style="background:#eef2ff;color:#4f46e5;border-radius:4px;padding:2px 7px;font-size:0.72rem;font-weight:700;">Sec. 2+3</span></td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;background:#fafbff;">
          <td style="padding:9px 12px;text-align:center;font-weight:700;color:#4f46e5;">2</td>
          <td style="padding:9px 12px;font-weight:600;color:#1e293b;">CLS + pos. encoding</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, 196, 768)</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, 197, 768)</td>
          <td style="padding:9px 12px;color:#374151;">Learnable CLS token prepended; learned 1D positional embeddings added element-wise</td>
          <td style="padding:9px 12px;text-align:center;"><span style="background:#eef2ff;color:#4f46e5;border-radius:4px;padding:2px 7px;font-size:0.72rem;font-weight:700;">Sec. 4</span></td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:9px 12px;text-align:center;font-weight:700;color:#4f46e5;">3</td>
          <td style="padding:9px 12px;font-weight:600;color:#1e293b;">Transformer encoder (×12)</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, 197, 768)</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, 197, 768)</td>
          <td style="padding:9px 12px;color:#374151;">12 identical blocks each run: LayerNorm &rarr; self-attention &rarr; residual &rarr; LayerNorm &rarr; FFN &rarr; residual</td>
          <td style="padding:9px 12px;text-align:center;"><span style="background:#eef2ff;color:#4f46e5;border-radius:4px;padding:2px 7px;font-size:0.72rem;font-weight:700;">Sec. 5+6</span></td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;background:#fafbff;">
          <td style="padding:9px 12px;text-align:center;font-weight:700;color:#4f46e5;">4</td>
          <td style="padding:9px 12px;font-weight:600;color:#1e293b;">Final LayerNorm</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, 197, 768)</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, 197, 768)</td>
          <td style="padding:9px 12px;color:#374151;">One last normalization applied to all tokens before the head</td>
          <td style="padding:9px 12px;text-align:center;"><span style="background:#eef2ff;color:#4f46e5;border-radius:4px;padding:2px 7px;font-size:0.72rem;font-weight:700;">Sec. 6</span></td>
        </tr>
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:9px 12px;text-align:center;font-weight:700;color:#4f46e5;">5</td>
          <td style="padding:9px 12px;font-weight:600;color:#1e293b;">CLS extraction + head</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, 197, 768)</td>
          <td style="padding:9px 12px;text-align:center;font-family:monospace;color:#6366f1;">(B, num_classes)</td>
          <td style="padding:9px 12px;color:#374151;">Token at position 0 (CLS) extracted &rarr; linear layer maps 768 dims to class scores (logits)</td>
          <td style="padding:9px 12px;text-align:center;"><span style="background:#eef2ff;color:#4f46e5;border-radius:4px;padding:2px 7px;font-size:0.72rem;font-weight:700;">Sec. 7</span></td>
        </tr>
      </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Zone 2: Model variants ───────────────────────────────────
    st.markdown("### ViT model family")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:12px;'>All standard ViT variants share the same architecture. "
        "What changes is depth (L), embedding dimension (D), and number of heads. "
        "Head dimension is always fixed at D/H&nbsp;=&nbsp;64. "
        "The highlighted row is what we built in this walkthrough.</p>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <table class="comp-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Layers (L)</th>
          <th>Embed dim (D)</th>
          <th>Heads (H)</th>
          <th>Head dim D/H</th>
          <th>Parameters</th>
          <th>IN-1k top-1 *</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ViT-Tiny/16</td>
          <td>12</td><td>192</td><td>3</td><td>64</td><td>5.7 M</td><td>~72%</td>
        </tr>
        <tr>
          <td>ViT-Small/16</td>
          <td>12</td><td>384</td><td>6</td><td>64</td><td>22 M</td><td>~81%</td>
        </tr>
        <tr style="background:#eef2ff;font-weight:700;">
          <td style="color:#4f46e5;">ViT-Base/16 &#9654; this walkthrough</td>
          <td style="color:#4f46e5;">12</td>
          <td style="color:#4f46e5;">768</td>
          <td style="color:#4f46e5;">12</td>
          <td style="color:#4f46e5;">64</td>
          <td style="color:#4f46e5;">86 M</td>
          <td style="color:#4f46e5;">~85%</td>
        </tr>
        <tr>
          <td>ViT-Large/16</td>
          <td>24</td><td>1024</td><td>16</td><td>64</td><td>307 M</td><td>~86%</td>
        </tr>
        <tr>
          <td>ViT-Huge/14</td>
          <td>32</td><td>1280</td><td>16</td><td>80</td><td>632 M</td><td>~88%</td>
        </tr>
      </tbody>
    </table>
    <p style="font-family:Inter,sans-serif;font-size:0.72rem;color:#94a3b8;margin-top:6px;">
    * Approximate ImageNet-1k top-1 accuracy after fine-tuning from ImageNet-21k pretraining.
    ViT-Huge uses patch size 14 and a slightly larger head dim.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("### What makes each variant different?")
    st.markdown("""
    <div class="enc-cards">
      <div class="enc-card" style="background:#f0f9ff;border-left:4px solid #0284c7;">
        <div class="enc-card-title" style="color:#0c4a6e;">Depth (number of layers L)</div>
        <div class="enc-card-body">
          ViT-Tiny/Small/Base all use L=12. ViT-Large doubles to 24; ViT-Huge uses 32.
          More layers = more rounds of attention and FFN refinement. Each layer sees
          the output of the previous one, building progressively more abstract features.
          Deeper models improve accuracy but increase training time linearly.
        </div>
      </div>
      <div class="enc-card" style="background:#fdf4ff;border-left:4px solid #a855f7;">
        <div class="enc-card-title" style="color:#7e22ce;">Embedding dimension D</div>
        <div class="enc-card-body">
          Goes from 192 (Tiny) to 1280 (Huge). Wider models can represent more
          information per token but cost more memory and compute.
          Every linear layer inside the block scales as D&sup2;, so doubling D
          quadruples the parameters in each block.
          D/H = 64 is fixed across all standard variants, so heads always operate
          in 64-dim space.
        </div>
      </div>
      <div class="enc-card" style="background:#fffbeb;border-left:4px solid #d97706;">
        <div class="enc-card-title" style="color:#92400e;">Patch size P</div>
        <div class="enc-card-body">
          All variants above use P=16 (196 patches). The /14 in ViT-Huge means P=14
          (256 patches), giving finer spatial resolution at the cost of a longer sequence.
          Smaller patches = more tokens = O(N&sup2;) more attention compute.
          P=32 variants (ViT-B/32, ViT-L/32) are faster but less accurate.
        </div>
      </div>
      <div class="enc-card" style="background:#f0fdf4;border-left:4px solid #16a34a;">
        <div class="enc-card-title" style="color:#14532d;">Choosing a variant</div>
        <div class="enc-card-body">
          For fine-tuning on a new task: <strong>ViT-Base/16</strong> is the default starting point.
          Strong accuracy, reasonable memory footprint (fits on a single 16 GB GPU).
          ViT-Large and ViT-Huge are for when you need the last 1-2% accuracy
          and have the compute budget. ViT-Tiny/Small are good for edge deployment
          or when data is limited.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Zone 3: Demo ─────────────────────────────────────────────
    st.markdown("### Run the complete forward pass")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:14px;'>Builds a full randomly-initialized ViT-Base and runs a "
        "real forward pass end to end. Watch the tensor flow through every stage.</p>",
        unsafe_allow_html=True
    )

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        batch_size = st.selectbox("Batch size", [1, 4, 8], index=0)
    with d_col2:
        n_classes = st.selectbox("Num classes", [10, 100, 1000], index=0)

    if st.button("Run full ViT forward pass", type="primary"):
        with st.spinner("Building ViT-Base and running forward pass..."):

            class _PatchEmb(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.proj = nn.Conv2d(3, 768, 16, 16)
                def forward(self, x):
                    return self.proj(x).flatten(2).transpose(1, 2)

            class _ViTEmb(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.pe  = _PatchEmb()
                    self.cls = nn.Parameter(torch.zeros(1, 1, 768))
                    self.pos = nn.Parameter(torch.zeros(1, 197, 768))
                def forward(self, x):
                    B = x.shape[0]
                    x = self.pe(x)
                    x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
                    return x + self.pos

            class _MHA(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.qkv  = nn.Linear(768, 768*3, bias=False)
                    self.proj = nn.Linear(768, 768)
                def forward(self, x):
                    B, N, D = x.shape
                    qkv = self.qkv(x).reshape(B,N,3,12,64).permute(2,0,3,1,4)
                    q, k, v = qkv.unbind(0)
                    a = (q @ k.transpose(-2,-1)) * (64**-0.5)
                    return self.proj((a.softmax(-1) @ v).transpose(1,2).reshape(B,N,D))

            class _Block(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.n1 = nn.LayerNorm(768); self.attn = _MHA()
                    self.n2 = nn.LayerNorm(768)
                    self.mlp = nn.Sequential(nn.Linear(768,3072),nn.GELU(),nn.Linear(3072,768))
                def forward(self, x):
                    x = x + self.attn(self.n1(x))
                    x = x + self.mlp(self.n2(x))
                    return x

            class _FullViT(nn.Module):
                def __init__(self, nc=10):
                    super().__init__()
                    self.emb  = _ViTEmb()
                    self.enc  = nn.Sequential(*[_Block() for _ in range(12)])
                    self.norm = nn.LayerNorm(768)
                    self.head = nn.Linear(768, nc)
                def forward(self, x):
                    x = self.emb(x)
                    x = self.enc(x)
                    x = self.norm(x)
                    return self.head(x[:, 0])

            model = _FullViT(nc=n_classes)
            x_in  = torch.randn(batch_size, 3, 224, 224)

            with torch.no_grad():
                # Capture intermediate shapes
                emb_out  = model.emb(x_in)
                enc_out  = model.enc(emb_out)
                norm_out = model.norm(enc_out)
                logits   = model.head(norm_out[:, 0])

            total = sum(p.numel() for p in model.parameters())

            # Shape walkthrough visual
            st.markdown("**Tensor shape at every stage:**")
            stages = [
                ("Input image",            tuple(x_in.shape),       "#f8fafc", "#4f46e5"),
                ("After patch embed",       (batch_size, 196, 768),  "#fdf4ff", "#a855f7"),
                ("After CLS + pos. enc.",   tuple(emb_out.shape),    "#fdf4ff", "#a855f7"),
                ("After 12 encoder blocks", tuple(enc_out.shape),    "#f0fdf4", "#16a34a"),
                ("After final LayerNorm",   tuple(norm_out.shape),   "#f0fdf4", "#16a34a"),
                ("CLS token extracted",     (batch_size, 768),       "#fffbeb", "#d97706"),
                ("Final logits",            tuple(logits.shape),     "#eef2ff", "#4f46e5"),
            ]
            cols = st.columns(len(stages))
            for col, (label, shape, bg, fg) in zip(cols, stages):
                col.markdown(
                    f"<div style='background:{bg};border:1px solid {fg}33;"
                    f"border-top:3px solid {fg};border-radius:6px;padding:8px 6px;"
                    f"text-align:center;'>"
                    f"<div style='font-family:Inter,sans-serif;font-size:0.65rem;"
                    f"font-weight:600;color:{fg};margin-bottom:4px;'>{label}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;"
                    f"color:#1e293b;font-weight:700;'>{shape}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown(f"""
            <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin:14px 0 6px;'>
            <strong>Total parameters: {total:,}</strong> &nbsp; (ViT-Base/16 reference: 86,415,592)
            </p>
            """, unsafe_allow_html=True)

            # Parameter breakdown table
            import pandas as pd
            st.markdown("**Parameter breakdown by component:**")
            bd_components = ["Patch embedding", "CLS token", "Positional encoding",
                             "12 Encoder blocks", "Final LayerNorm", "MLP Head"]
            bd_params = [
                sum(p.numel() for p in model.emb.pe.parameters()),
                model.emb.cls.numel(),
                model.emb.pos.numel(),
                sum(p.numel() for p in model.enc.parameters()),
                sum(p.numel() for p in model.norm.parameters()),
                sum(p.numel() for p in model.head.parameters()),
            ]
            df_bd = pd.DataFrame({
                "Component": bd_components,
                "Parameters": [f"{p:,}" for p in bd_params],
                "% of total": [f"{p/total*100:.1f}%" for p in bd_params],
            })
            st.dataframe(df_bd, hide_index=True, use_container_width=True)

            # Mini bar chart of breakdown
            fig, ax = plt.subplots(figsize=(9, 3))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            bar_cols = ['#4f46e5','#7c3aed','#0284c7','#059669','#d97706','#dc2626']
            bars = ax.barh(bd_components, bd_params, color=bar_cols,
                           edgecolor='#f1f5f9', linewidth=0.5)
            for bar, val in zip(bars, bd_params):
                ax.text(bar.get_width() + total*0.005, bar.get_y()+bar.get_height()/2,
                        f"{val/1e6:.1f}M", va='center', fontsize=8, color='#374151')
            ax.set_xlabel("Parameters", color='#475569', fontsize=9)
            ax.tick_params(colors='#64748b', labelsize=8)
            ax.spines[:].set_color('#e2e8f0')
            ax.set_title("Where the 86M parameters live", color='#0f172a', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        info("Click the button to build ViT-Base from scratch and trace a tensor through every stage.")

    # ── Zone 4: Code (collapsed) ─────────────────────────────────
    with st.expander("Show the complete ViT implementation", expanded=False):
        st.code("""
class ViT(nn.Module):
    def __init__(
        self,
        image_size=224, patch_size=16,
        num_classes=1000, embed_dim=768,
        depth=12, num_heads=12,
        mlp_ratio=4.0, dropout=0.1
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2   # 196

        # Stage 1: patch embedding + CLS + positional encoding
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.dropout     = nn.Dropout(dropout)

        # Stage 2: stack of transformer encoder blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Stage 3: final LayerNorm + classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding: (B,3,224,224) -> (B,196,768)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Prepend CLS token and add positional encoding
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)        # (B, 197, 768)
        x   = self.dropout(x + self.pos_embed)

        # Transformer encoder
        x = self.blocks(x)                       # (B, 197, 768)
        x = self.norm(x)

        # Classification head: CLS token only
        return self.head(x[:, 0])                # (B, num_classes)


# ViT-Base/16
model = ViT(image_size=224, patch_size=16, num_classes=1000,
            embed_dim=768, depth=12, num_heads=12)
total = sum(p.numel() for p in model.parameters())
# total = 86,415,592
        """, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif section.startswith("9"):
    section_header("9", "Training ViT",
                   "Data requirements, training recipe, and fine-tuning strategy")

    # ── Zone 1: Why ViT training is different ───────────────────
    st.markdown("### Why ViT training is different from CNNs")
    st.markdown("""
    <div class="pretrained-card" style="margin-bottom:18px;">
    <p style="font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin:0 0 10px;">
    Convolutional networks have <strong>inductive biases</strong> baked in: locality (a filter only
    looks at a small neighborhood) and translation equivariance (the same filter applies everywhere).
    These biases let CNNs learn effectively from tens of thousands of images.
    </p>
    <p style="font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin:0;">
    ViT has <strong>no such biases</strong>. Self-attention can connect any two patches from the first
    layer. This generality is ViT's strength on large datasets, but it means the model must
    learn spatial structure from scratch; it needs far more data to do so. The original paper
    showed ViT underperforms ResNets when trained only on ImageNet-1k (1.2M images), but
    surpasses them after pretraining on ImageNet-21k (14M) or JFT-300M (300M).
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Three training scenarios")
    st.markdown("""
    <div class="enc-cards">
      <div class="enc-card" style="background:#fffbeb;border-left:4px solid #d97706;">
        <div class="enc-card-title" style="color:#92400e;">A. Fine-tune a pretrained ViT</div>
        <div class="enc-card-body">
          <strong>When:</strong> You have a custom dataset (even just a few thousand images).<br><br>
          <strong>What:</strong> Download a ViT-Base/16 pretrained on ImageNet-21k.
          Replace the classification head with one matching your number of classes.
          The encoder weights already encode rich visual features; you just redirect them.<br><br>
          <strong>Result:</strong> Competitive accuracy in 10-30 epochs, no massive compute needed.
          This is the recommended path for most real tasks.
        </div>
      </div>
      <div class="enc-card" style="background:#f0fdf4;border-left:4px solid #16a34a;">
        <div class="enc-card-title" style="color:#14532d;">B. Feature extraction (fastest)</div>
        <div class="enc-card-body">
          <strong>When:</strong> Very small dataset or limited compute. The pretrained ViT is
          used as a frozen feature extractor; only the new classification head is trained.<br><br>
          <strong>What:</strong> Freeze all encoder weights
          (<code>requires_grad = False</code>), then train just the linear head.<br><br>
          <strong>Result:</strong> Trains in minutes. Lower accuracy than full fine-tuning,
          but zero risk of degrading the pretrained features.
        </div>
      </div>
      <div class="enc-card" style="background:#fdf4ff;border-left:4px solid #a855f7;">
        <div class="enc-card-title" style="color:#7e22ce;">C. Train from scratch</div>
        <div class="enc-card-body">
          <strong>When:</strong> You have millions of images and significant GPU budget.<br><br>
          <strong>What:</strong> Initialize all weights randomly and train for 90-300 epochs
          with heavy augmentation (RandAugment, CutMix, MixUp). Use a large batch size
          (1024+) and a multi-GPU setup.<br><br>
          <strong>Result:</strong> Can match or exceed CNN performance but requires
          infrastructure most students won't have access to.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Zone 2: Key training ingredients ────────────────────────
    st.markdown("### Key training ingredients")
    st.markdown("""
    <table class="comp-table" style="table-layout:fixed;width:100%;">
      <colgroup>
        <col style="width:16%;">
        <col style="width:22%;">
        <col style="width:62%;">
      </colgroup>
      <thead>
        <tr>
          <th>Ingredient</th>
          <th>Typical value</th>
          <th style="text-align:left;">Why it matters for ViT</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Optimizer</strong></td>
          <td><code>AdamW</code>, lr=1e-3</td>
          <td style="text-align:left;">Adam with decoupled weight decay. ViT is sensitive to weight decay;
              use 0.1 for training from scratch, 0.01 for fine-tuning.</td>
        </tr>
        <tr>
          <td><strong>LR schedule</strong></td>
          <td>Cosine decay + linear warmup (5-10 epochs)</td>
          <td style="text-align:left;">Warmup prevents large initial updates from damaging pretrained weights.
              Cosine decay gives a smooth ramp-down to zero.</td>
        </tr>
        <tr>
          <td><strong>Gradient clipping</strong></td>
          <td><code>max_norm=1.0</code></td>
          <td style="text-align:left;">ViT can produce occasional large gradients, especially at the start.
              Clipping prevents these from destabilizing training without affecting
              normal gradient updates.</td>
        </tr>
        <tr>
          <td><strong>Label smoothing</strong></td>
          <td>0.1</td>
          <td style="text-align:left;">Replaces one-hot targets with 0.9 on the correct class and 0.1/N spread
              across all others. Reduces overconfidence; ViT benefits more than CNNs.</td>
        </tr>
        <tr>
          <td><strong>Data augmentation</strong></td>
          <td>RandomResizedCrop, RandomFlip, ColorJitter</td>
          <td style="text-align:left;">ViT has no translation equivariance, so augmentation is the main source
              of positional robustness. RandAugment and CutMix improve accuracy
              significantly on small datasets.</td>
        </tr>
        <tr>
          <td><strong>Layerwise LR decay</strong></td>
          <td>decay=0.75 per layer (fine-tuning only)</td>
          <td style="text-align:left;">Lower encoder layers learn general features (edges, textures) that transfer
              well. They need smaller LR updates to preserve that knowledge.
              Higher layers are more task-specific and can tolerate larger updates.</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Zone 3: Fine-tuning walkthrough ─────────────────────────
    st.markdown("### Fine-tuning walkthrough (recommended path)")
    st.markdown("""
    <div style="font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin-bottom:14px;">
    The four-step process below covers the most common real-world use case:
    take a pretrained ViT-Base/16 and adapt it to a new classification task.
    </div>
    """, unsafe_allow_html=True)

    steps_col1, steps_col2 = st.columns(2)
    with steps_col1:
        st.markdown("""
        <div class="math-box" style="margin-bottom:12px;">
          <div class="math-step">
            <div class="math-step-num" style="display:flex;align-items:center;justify-content:center;flex-shrink:0;">1</div>
            <div>
              <strong>Load pretrained model</strong><br>
              <span style="font-size:0.80rem;color:#475569;">
              Download ViT-Base/16 pretrained on ImageNet-21k from HuggingFace.
              The encoder already knows edges, textures, and high-level visual concepts.
              </span>
            </div>
          </div>
          <div class="math-step">
            <div class="math-step-num" style="display:flex;align-items:center;justify-content:center;flex-shrink:0;">2</div>
            <div>
              <strong>Replace the classification head</strong><br>
              <span style="font-size:0.80rem;color:#475569;">
              Swap the 1000-class head for a new linear layer sized to your task
              (e.g., 10 for CIFAR-10). Only this layer starts with random weights.
              </span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with steps_col2:
        st.markdown("""
        <div class="math-box" style="margin-bottom:12px;">
          <div class="math-step">
            <div class="math-step-num" style="display:flex;align-items:center;justify-content:center;flex-shrink:0;">3</div>
            <div>
              <strong>Phase 1: warm up the head (optional)</strong><br>
              <span style="font-size:0.80rem;color:#475569;">
              Freeze the encoder and train only the new head for 5 epochs with lr=1e-3.
              This prevents a randomly-initialized head from corrupting the encoder
              weights with large early gradients.
              </span>
            </div>
          </div>
          <div class="math-step">
            <div class="math-step-num" style="display:flex;align-items:center;justify-content:center;flex-shrink:0;">4</div>
            <div>
              <strong>Phase 2: full fine-tuning with layerwise decay</strong><br>
              <span style="font-size:0.80rem;color:#475569;">
              Unfreeze all layers. Use a lower base LR (1e-4) and apply
              0.75x decay per layer so early encoder layers update more slowly
              than the head.
              </span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Zone 4: Demo ─────────────────────────────────────────────
    st.markdown("### Dry-run: full forward + backward pass")
    st.markdown(
        "<p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;"
        "margin-bottom:14px;'>Builds a depth-4 ViT (same architecture as ViT-Base but "
        "4 encoder layers instead of 12 to run faster), runs one complete training step, "
        "and reports the loss and gradient norm. Confirms the full computational graph "
        "works end to end.</p>",
        unsafe_allow_html=True
    )

    nc = st.selectbox("Num classes for dry run", [10, 100, 1000], index=0)

    if st.button("Run dry-run training step", type="primary"):
        with st.spinner("Building model, running forward + backward pass..."):
            try:
                class _TinyViT(nn.Module):
                    def __init__(self, nc=10, depth=4):
                        super().__init__()
                        self.proj = nn.Conv2d(3, 768, 16, 16)
                        self.cls  = nn.Parameter(torch.zeros(1, 1, 768))
                        self.pos  = nn.Parameter(torch.zeros(1, 197, 768))
                        self.enc  = nn.ModuleList([
                            nn.TransformerEncoderLayer(
                                d_model=768, nhead=12,
                                dim_feedforward=3072,
                                batch_first=True, norm_first=True
                            ) for _ in range(depth)
                        ])
                        self.norm = nn.LayerNorm(768)
                        self.head = nn.Linear(768, nc)

                    def forward(self, x):
                        B = x.shape[0]
                        x = self.proj(x).flatten(2).transpose(1, 2)
                        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
                        x = x + self.pos
                        for blk in self.enc:
                            x = blk(x)
                        return self.head(self.norm(x)[:, 0])

                m_dry    = _TinyViT(nc=nc, depth=4)
                opt_dry  = torch.optim.AdamW(m_dry.parameters(), lr=1e-3, weight_decay=0.1)
                crit_dry = nn.CrossEntropyLoss(label_smoothing=0.1)

                x_dry      = torch.randn(4, 3, 224, 224)
                labels_dry = torch.randint(0, nc, (4,))

                m_dry.train()
                opt_dry.zero_grad()
                logits_dry = m_dry(x_dry)
                loss_dry   = crit_dry(logits_dry, labels_dry)
                loss_dry.backward()
                grad_norm  = torch.nn.utils.clip_grad_norm_(m_dry.parameters(), 1.0)
                opt_dry.step()

                total_dry = sum(p.numel() for p in m_dry.parameters())
                st.success("Forward + backward pass completed.")

                # Results in colored stage cards
                result_data = [
                    ("Input shape",    str(tuple(x_dry.shape)),      "#eef2ff", "#4f46e5"),
                    ("Output shape",   str(tuple(logits_dry.shape)), "#eef2ff", "#4f46e5"),
                    ("Loss",           f"{loss_dry.item():.4f}",     "#fef3c7", "#d97706"),
                    ("Gradient norm",  f"{grad_norm.item():.4f}",    "#dcfce7", "#16a34a"),
                    ("Parameters",     f"{total_dry:,}",             "#f0f9ff", "#0284c7"),
                    ("Encoder depth",  "4 (ViT-Base = 12)",          "#fdf4ff", "#a855f7"),
                ]
                r_cols = st.columns(len(result_data))
                for col, (label, val, bg, fg) in zip(r_cols, result_data):
                    col.markdown(
                        f"<div style='background:{bg};border:1px solid {fg}33;"
                        f"border-top:3px solid {fg};border-radius:6px;padding:8px 6px;"
                        f"text-align:center;'>"
                        f"<div style='font-family:Inter,sans-serif;font-size:0.65rem;"
                        f"font-weight:600;color:{fg};margin-bottom:4px;'>{label}</div>"
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;"
                        f"color:#1e293b;font-weight:700;'>{val}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                success("""
                Every stage of the computational graph executed correctly:
                patch projection, CLS token, positional encoding, encoder stack,
                CLS extraction, linear head, cross-entropy loss with label smoothing,
                backpropagation, gradient clipping, and AdamW weight update.
                To run the full ViT-Base, change <code>depth=4</code> to <code>depth=12</code>.
                """)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        info("Click the button to run one complete forward + backward training step.")

    st.divider()

    # ── Zone 5: Code (collapsed) ─────────────────────────────────
    with st.expander("Show the fine-tuning recipe", expanded=False):
        st.code("""
from transformers import ViTForImageClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

# Step 1: load pretrained ViT-Base/16 from HuggingFace
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k'
)

# Step 2: replace the head for our task
model.classifier = nn.Linear(
    model.config.hidden_size,   # 768
    10                          # CIFAR-10 classes
)

# Step 3 (optional): warm up the head -- freeze encoder first
for param in model.vit.parameters():
    param.requires_grad = False

optimizer = AdamW(model.classifier.parameters(), lr=1e-3)
# Train 5 epochs; the head converges quickly

# Step 4: full fine-tune with layerwise LR decay
for param in model.parameters():
    param.requires_grad = True

decay = 0.75   # each earlier layer gets 0.75x the LR of the one above it
param_groups = []
for i, layer in enumerate(model.vit.encoder.layer):   # 12 layers
    lr_scale = decay ** (12 - i)
    param_groups.append({'params': layer.parameters(), 'lr': 1e-4 * lr_scale})
param_groups.append({'params': model.classifier.parameters(), 'lr': 1e-4})

optimizer  = AdamW(param_groups, weight_decay=0.01)
scheduler  = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

def train_epoch(loader):
    model.train()
    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        loss = criterion(model(images).logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
        """, language="python")

    with st.expander("Show the from-scratch training recipe", expanded=False):
        st.code("""
from torchvision import transforms, datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

# Heavy augmentation is essential for training from scratch
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

train_set  = datasets.CIFAR10('./data', train=True,  transform=train_transform, download=True)
val_set    = datasets.CIFAR10('./data', train=False, transform=val_transform,   download=True)

model      = ViT(num_classes=10).cuda()
optimizer  = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
scheduler  = CosineAnnealingLR(optimizer, T_max=90, eta_min=1e-5)
criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

def train_epoch(loader):
    model.train()
    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
        """, language="python")

    with st.expander("Complete Colab fine-tuning notebook (CIFAR-10 or your own dataset)", expanded=False):
        st.markdown("""
        <p style='font-family:Inter,sans-serif;font-size:0.875rem;color:#374151;margin:0 0 10px;'>
        Copy each cell into a new Colab notebook. Change <code>DATASET</code> at the top of
        Cell 2 to switch between CIFAR-10 (zero setup, runs immediately) and your own images
        (requires a Google Drive folder in the standard ImageFolder layout).
        </p>
        <p style='font-family:Inter,sans-serif;font-size:0.80rem;color:#6366f1;margin:0 0 14px;'>
        <strong>Custom dataset folder layout on Google Drive:</strong><br>
        <code>MyDrive/my_dataset/train/cat/img1.jpg ...</code><br>
        <code>MyDrive/my_dataset/train/dog/img1.jpg ...</code><br>
        <code>MyDrive/my_dataset/val/cat/img1.jpg ...</code><br>
        <code>MyDrive/my_dataset/val/dog/img1.jpg ...</code>
        </p>
        """, unsafe_allow_html=True)
        st.code("""# ── Cell 1: Install dependencies ──────────────────────────────
!pip install transformers timm -q
# torch, torchvision, pillow already present in Colab
""", language="python")

        st.code("""# ── Cell 2: Configuration  (edit this cell only) ──────────────

DATASET     = "cifar10"        # "cifar10"  or  "custom"
NUM_CLASSES = 10               # number of classes in your dataset
NUM_EPOCHS  = 20               # fine-tuning epochs
BATCH_SIZE  = 32
LR_HEAD     = 1e-3             # head learning rate (phase 1 + 2)
LR_ENCODER  = 1e-4             # encoder base LR (phase 2)
LR_DECAY    = 0.75             # layerwise LR decay factor
WARMUP_EP   = 3                # linear warmup epochs

# Only used when DATASET = "custom"
DRIVE_PATH  = "/content/drive/MyDrive/my_dataset"
""", language="python")

        st.code("""# ── Cell 3: Mount Drive (only needed for custom dataset) ──────
if DATASET == "custom":
    from google.colab import drive
    drive.mount("/content/drive")
""", language="python")

        st.code("""# ── Cell 4: Data loaders ──────────────────────────────────────
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

if DATASET == "cifar10":
    train_set = datasets.CIFAR10("./data", train=True,  transform=train_tf, download=True)
    val_set   = datasets.CIFAR10("./data", train=False, transform=val_tf,   download=True)
    CLASS_NAMES = train_set.classes
else:
    train_set   = datasets.ImageFolder(f"{DRIVE_PATH}/train", transform=train_tf)
    val_set     = datasets.ImageFolder(f"{DRIVE_PATH}/val",   transform=val_tf)
    CLASS_NAMES = train_set.classes
    NUM_CLASSES = len(CLASS_NAMES)    # override if not set above

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"Train: {len(train_set):,} images  |  Val: {len(val_set):,} images")
""", language="python")

        st.code("""# ── Cell 5: Load pretrained ViT and replace head ──────────────
import torch.nn as nn
from transformers import ViTForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,   # replaces the pretrained head automatically
)
model = model.to(device)

total  = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}  |  Classes: {NUM_CLASSES}")
""", language="python")

        st.code("""# ── Cell 6: Phase 1 -- train head only (freeze encoder) ────────
from torch.optim import AdamW

for param in model.vit.parameters():
    param.requires_grad = False

optimizer = AdamW(model.classifier.parameters(), lr=LR_HEAD, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def run_epoch(loader, train=True):
    model.train(train)
    total_loss, correct, n = 0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(imgs).logits
            loss = criterion(out, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            n          += labels.size(0)
    return total_loss / n, 100.0 * correct / n

print("Phase 1: head warmup (encoder frozen)")
for ep in range(WARMUP_EP):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    print(f"  ep {ep+1}/{WARMUP_EP}  train {tr_acc:.1f}%  val {vl_acc:.1f}%  loss {tr_loss:.4f}")
""", language="python")

        st.code("""# ── Cell 7: Phase 2 -- full fine-tune with layerwise LR decay ──
from torch.optim.lr_scheduler import CosineAnnealingLR

for param in model.parameters():
    param.requires_grad = True

# Lower layers get smaller LR: layer 0 gets LR_ENCODER * LR_DECAY^11
param_groups = []
for i, layer in enumerate(model.vit.encoder.layer):
    scale = LR_DECAY ** (len(model.vit.encoder.layer) - 1 - i)
    param_groups.append({"params": layer.parameters(), "lr": LR_ENCODER * scale})
# Embeddings + LayerNorm get the smallest LR
param_groups.append({"params": model.vit.embeddings.parameters(), "lr": LR_ENCODER * (LR_DECAY ** 12)})
param_groups.append({"params": model.vit.layernorm.parameters(),  "lr": LR_ENCODER})
param_groups.append({"params": model.classifier.parameters(),     "lr": LR_HEAD})

optimizer = AdamW(param_groups, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

print(f"Phase 2: full fine-tune for {NUM_EPOCHS} epochs")
best_val_acc = 0.0
for ep in range(NUM_EPOCHS):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    scheduler.step()
    marker = " <-- best" if vl_acc > best_val_acc else ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), "best_vit.pt")
    print(f"  ep {ep+1:02d}/{NUM_EPOCHS}  train {tr_acc:.1f}%  val {vl_acc:.1f}%  loss {tr_loss:.4f}{marker}")

print(f"Best val accuracy: {best_val_acc:.1f}%  (saved to best_vit.pt)")
""", language="python")

        st.code("""# ── Cell 8: Quick evaluation on a single image ────────────────
from PIL import Image
import requests, torch
from torchvision import transforms

# Load best checkpoint
model.load_state_dict(torch.load("best_vit.pt", map_location=device))
model.eval()

# Replace this URL with any image, or use: Image.open("/path/to/img.jpg")
url   = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
x = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x).logits
    probs  = logits.softmax(-1)[0]

top5 = probs.topk(min(5, NUM_CLASSES))
print("Top predictions:")
for prob, idx in zip(top5.values, top5.indices):
    print(f"  {CLASS_NAMES[idx]:20s}  {prob.item()*100:.1f}%")
""", language="python")
