"""
Vision Transformer (ViT): From Pixels to Prediction
SEAS 8525 - Computer Vision and Generative AI
Dr. Elbasheer

Run:  streamlit run vit_app.py
Theme: config.toml lives in the .streamlit folder next to this file.

Dosovitskiy et al. (2020), "An Image is Worth 16x16 Words".
"""

import math

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

st.set_page_config(
    page_title="ViT Walkthrough",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Palette (one colour code for the whole app) ------------------------------
NAVY   = "#002147"
GOLD   = "#FFC400"
BLUE   = "#1D4ED8"   # patches and inputs
PURPLE = "#7C3AED"   # queries / keys / values
RED    = "#E11D48"   # the CLS token
GREEN  = "#0E9F6E"   # outputs and predictions
ORANGE = "#D97706"   # position embeddings and training signal
MUTED  = "#475569"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root{
  --navy:#002147; --gold:#FFC400; --bg:#F3F6FB; --border:#D9E1EC;
  --text:#0F172A; --muted:#475569;
  --blue:#1D4ED8; --purple:#7C3AED; --red:#E11D48; --green:#0E9F6E; --orange:#D97706;
}

/* ── App shell ── */
.stApp{ background:var(--bg); color:var(--text); }
[data-testid="stHeader"]{ background:transparent; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp div, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3{
  font-family:'Lexend', system-ui, sans-serif;
}
.stApp code, .stApp pre, .stApp pre *{ font-family:'JetBrains Mono', monospace !important; }
[data-testid="stMainBlockContainer"], .block-container{
  max-width:1180px; padding-top:2.2rem; padding-bottom:3rem;
}
[data-testid="stSidebar"]{ background:#FFFFFF; border-right:1px solid var(--border); }

/* ── Widgets ── */
.stApp label p{ font-size:1rem !important; font-weight:600; color:var(--navy); }
.stTextInput input, .stTextArea textarea{
  font-size:1.05rem !important; border-radius:10px !important;
}
.stButton > button{ border-radius:10px; font-weight:700; font-size:1rem; padding:.55rem 1.3rem; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--navy) !important; border:none !important; color:#fff !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
  background:#0B3470 !important; color:var(--gold) !important;
}
[data-testid="stExpander"] details{
  background:#fff; border:1px solid var(--border) !important; border-radius:14px !important;
}
[data-testid="stExpander"] summary p{ font-weight:700; color:var(--navy); font-size:1.08rem; }
[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:16px; }

/* Sidebar radio as a clean menu */
[data-testid="stSidebar"] [role="radiogroup"]{ gap:2px; }
[data-testid="stSidebar"] [role="radiogroup"] label{
  padding:8px 10px; border-radius:10px; width:100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{ background:#F3F6FB; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){ background:#EAF0FA; }
[data-testid="stSidebar"] [role="radiogroup"] label p{
  font-size:.95rem !important; font-weight:500; color:#1E293B;
}

/* ── Section header ── */
.sec-hdr{ margin:0 0 22px; }
.sec-pill{ display:inline-block; background:var(--gold); color:var(--navy); font-weight:800;
  font-size:.78rem; letter-spacing:.12em; text-transform:uppercase;
  padding:5px 13px; border-radius:999px; }
.sec-title{ font-size:2.55rem; font-weight:800; color:var(--navy); letter-spacing:-.02em;
  line-height:1.12; margin:14px 0 8px; }
.sec-sub{ font-size:1.18rem; color:var(--muted); line-height:1.5; }
.sec-rule{ height:5px; width:80px; background:var(--gold); border-radius:4px; margin-top:16px; }

/* ── Main idea ── */
.key{ background:var(--navy); border-left:9px solid var(--gold); border-radius:18px;
  padding:24px 30px; margin:4px 0 28px; box-shadow:0 10px 26px rgba(0,33,71,.16); }
.key-label{ color:var(--gold); font-size:.8rem; font-weight:800; letter-spacing:.16em;
  text-transform:uppercase; }
.key-text{ color:#fff; font-size:1.45rem; font-weight:600; line-height:1.45; margin-top:8px; }
.key-text b{ color:var(--gold); font-weight:700; }
.key.hero{ padding:32px 38px; border-left-width:12px; margin:8px 0 26px; }
.key.hero .key-text{ font-size:1.85rem; line-height:1.4; margin-top:10px; }
@media (max-width:900px){
  .key.hero{ padding:24px 26px; }
  .key.hero .key-text{ font-size:1.45rem; }
}

/* ── Body text ── */
.lead{ font-size:1.2rem; line-height:1.75; color:#1E293B; margin:0 0 22px; }
.lead b{ color:var(--navy); }
.hint{ font-size:1.08rem; color:var(--muted); line-height:1.6; margin:0 0 14px; }
.sub{ display:flex; align-items:center; gap:13px; font-size:1.72rem; font-weight:800;
  color:var(--navy); margin:38px 0 16px; letter-spacing:-.01em; }
.sub::before{ content:""; width:10px; height:28px; background:var(--gold); border-radius:3px; }

/* ── Grids and cards ── */
.grid{ display:grid; gap:18px; margin:4px 0 20px; }
.g2{ grid-template-columns:repeat(2,1fr); }
.g3{ grid-template-columns:repeat(3,1fr); }
@media (max-width:900px){ .g2,.g3{ grid-template-columns:1fr; } }
.card{ background:#fff; border:1px solid var(--border); border-top:6px solid var(--c);
  border-radius:16px; padding:18px 22px; }
.card-title{ font-size:1.28rem; font-weight:700; color:var(--c); margin-bottom:8px; }
.card-body{ font-size:1.09rem; line-height:1.68; color:#334155; }
.card-body b{ color:var(--text); }
.card code{ font-size:.9rem; background:#F1F5F9; padding:2px 6px; border-radius:6px; color:var(--navy); }

/* ── Tags ── */
.tag{ display:inline-block; padding:3px 11px; border-radius:999px; font-size:.84rem;
  font-weight:700; color:var(--c); background:color-mix(in srgb, var(--c) 13%, white);
  border:1px solid color-mix(in srgb, var(--c) 35%, white); margin:2px 4px 2px 0;
  white-space:nowrap; }

/* ── Class tip ── */
.tip{ background:#FFFBEA; border:1px solid #FDE68A; border-left:8px solid var(--gold);
  border-radius:14px; padding:18px 24px; margin:24px 0; font-size:1.1rem; line-height:1.72;
  color:#1E293B; }
.tip-label{ display:inline-block; background:var(--navy); color:var(--gold); font-size:.72rem;
  font-weight:800; letter-spacing:.12em; text-transform:uppercase; padding:3px 10px;
  border-radius:999px; margin-right:8px; }

/* ── Tables ── */
.tbl-wrap{ background:#fff; border:1px solid var(--border); border-radius:16px;
  overflow:hidden; overflow-x:auto; margin:4px 0 20px; box-shadow:0 2px 8px rgba(0,33,71,.05); }
.tbl{ width:100%; border-collapse:collapse; font-size:1.08rem; }
.tbl th{ background:var(--navy); color:#fff; font-weight:700; text-align:left;
  padding:14px 18px; font-size:.95rem; letter-spacing:.02em; }
.tbl td{ padding:14px 18px; border-top:1px solid #E8EDF4; vertical-align:top;
  line-height:1.55; color:#1E293B; }
.tbl tbody tr:nth-child(even) td{ background:#F7F9FC; }
.tbl td:first-child{ font-weight:700; color:var(--navy); }
.tbl tr.hl td{ background:#FFF6D1 !important; }
.tbl td.ncol{ font-family:'JetBrains Mono', monospace; font-weight:700; text-align:center; }
.tbl th.ncol{ text-align:center; }
.tbl .mono{ font-family:'JetBrains Mono', monospace; font-size:.92rem; color:var(--navy);
  white-space:nowrap; }

/* ── Token pills ── */
.tok-row{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:12px 0; }
.tok{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1rem;
  padding:6px 12px; border-radius:10px; border:2px solid; line-height:1.3; }
.tok.cls { background:var(--gold); color:var(--navy); border-color:var(--gold); }
.tok.sep { background:#E2E8F0; color:var(--navy); border-color:#94A3B8; }
.tok.mask{ background:#FFE4EA; color:var(--red); border-color:var(--red); }
.tok.a   { background:#DBEAFE; color:var(--blue); border-color:#93C5FD; }
.tok.b   { background:#EDE9FE; color:var(--purple); border-color:#C4B5FD; }
.tok.out { background:#D1FAE5; color:var(--green); border-color:var(--green); }
.tok.wp  { border-style:dashed; }
.tok-legend{ display:flex; flex-wrap:wrap; gap:16px; font-size:.92rem; color:var(--muted);
  margin:6px 0 16px; }
.tok-legend span.sw{ display:inline-block; width:14px; height:14px; border-radius:4px;
  margin-right:6px; vertical-align:-2px; border:2px solid; }

/* ── Colored key-term bullets ── */
.blist{ margin:4px 0 18px; padding:0; list-style:none; }
.blist li{ position:relative; padding:12px 0 12px 32px; font-size:1.13rem; line-height:1.7;
  color:#334155; border-bottom:1px dashed var(--border); }
.blist li:last-child{ border-bottom:none; }
.blist li::before{ content:""; position:absolute; left:8px; top:21px; width:11px; height:11px;
  border-radius:50%; background:var(--c); }
.blist .k{ font-weight:800; color:var(--c); font-size:1.17rem; }
.blist b{ color:var(--navy); }
.blist code{ font-family:'JetBrains Mono', monospace; font-size:.89rem; background:#F1F5F9;
  padding:2px 6px; border-radius:6px; color:var(--navy); }
.blist .chain{ display:block; font-family:'JetBrains Mono', monospace; font-size:.9rem;
  color:var(--navy); background:#F7F9FC; border-radius:8px; padding:7px 11px; margin:7px 0 0; }

/* ── Numbered steps ── */
.steps{ background:#fff; border:1px solid var(--border); border-radius:16px; padding:6px 22px; }
.step{ display:flex; gap:16px; align-items:flex-start; padding:14px 0; }
.step + .step{ border-top:1px dashed var(--border); }
.num{ flex:0 0 34px; height:34px; border-radius:50%; background:var(--gold); color:var(--navy);
  font-weight:800; display:flex; align-items:center; justify-content:center; font-size:1rem; }
.step-text{ font-size:1.13rem; line-height:1.6; color:#1E293B; padding-top:4px; }
.step-text b{ color:var(--navy); }

/* ── Flow (ViT bridge) ── */
.flow{ display:flex; align-items:stretch; gap:14px; margin:10px 0 32px; flex-wrap:wrap; }
.flow-box{ flex:1 1 265px; background:#fff; border:1px solid var(--border);
  border-top:8px solid var(--c); border-radius:18px; padding:24px 26px 22px;
  box-shadow:0 4px 16px rgba(0,33,71,.07); }
.flow-kicker{ font-size:.82rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase;
  color:var(--c); }
.flow-title{ font-size:1.5rem; font-weight:800; color:var(--navy); margin:8px 0 14px;
  letter-spacing:-.01em; line-height:1.2; }
.flow-box ul{ margin:0; padding-left:20px; font-size:1.08rem; line-height:1.95; color:#334155; }
.flow-box ul b{ color:var(--navy); }
.flow-op{ display:flex; align-items:center; font-size:2.9rem; font-weight:800; color:var(--navy);
  padding:0 4px; }
@media (max-width:900px){ .flow-op{ justify-content:center; font-size:2.2rem; } }

/* ── Equation boxes (worked example) ── */
.eq{ display:flex; align-items:stretch; gap:10px; flex-wrap:wrap; margin:6px 0 10px; }
.eq-box{ flex:1 1 170px; border-radius:14px; padding:16px 16px; text-align:center;
  background:color-mix(in srgb, var(--c) 10%, white); border:2px solid var(--c); }
.eq-name{ font-weight:800; font-size:1.1rem; color:var(--c); }
.eq-desc{ font-size:.95rem; color:#334155; margin-top:6px; line-height:1.45; }
.eq-op{ display:flex; align-items:center; font-size:2rem; font-weight:800; color:var(--navy); }
.eq-note{ text-align:center; font-size:1rem; color:var(--muted); margin:6px 0 4px; }

/* ── Worked embedding example (worked example) ── */
.eg-wrap{ background:#fff; border:1px solid var(--border); border-radius:16px; padding:18px 18px;
  overflow-x:auto; margin:4px 0 8px; }
.eg{ display:grid; gap:8px 6px; align-items:center; min-width:820px; }
.eg-label{ font-weight:800; font-size:.98rem; text-align:right; padding-right:10px; }
.eg-col{ display:flex; justify-content:center; }
.eg-col .tok{ font-size:.85rem; padding:5px 8px; }
.eg-op{ font-size:1.3rem; font-weight:800; color:var(--navy); text-align:right; padding-right:14px;
  line-height:.8; }
.chip{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:.85rem; text-align:center;
  padding:7px 2px; border-radius:9px; color:var(--c); border:2px solid var(--c);
  background:color-mix(in srgb, var(--c) 10%, white); }

/* ── Vertical pipeline (flow stages) ── */
.pipe{ margin:8px 0 26px; }
.pipe-row{ display:flex; gap:16px; align-items:flex-start; }
.pipe-num{ flex:0 0 42px; height:42px; border-radius:50%; background:var(--c); color:#fff;
  font-weight:800; font-size:1.1rem; display:flex; align-items:center; justify-content:center;
  box-shadow:0 3px 10px rgba(0,33,71,.14); }
.pipe-body{ flex:1; min-width:0; background:#fff; border:1px solid var(--border);
  border-left:6px solid var(--c); border-radius:14px; padding:15px 22px; }
.pipe-what{ font-size:1.32rem; font-weight:800; color:var(--navy); letter-spacing:-.01em; }
.pipe-why{ font-size:1.1rem; line-height:1.72; color:#334155; margin-top:6px; }
.pipe-why b{ color:var(--navy); }
.pipe-why code{ font-family:'JetBrains Mono', monospace; font-size:.9rem; background:#F1F5F9;
  padding:2px 7px; border-radius:6px; color:var(--navy); }
.pipe-io{ font-family:'JetBrains Mono', monospace; font-size:.98rem; line-height:1.9;
  color:var(--navy); background:#F7F9FC; border-radius:10px; padding:10px 14px;
  margin:4px 0 10px; }
.pipe-io b{ color:var(--blue); }
.pipe-link{ width:42px; text-align:center; font-size:1.5rem; color:#94A3B8;
  line-height:1.5; font-weight:700; }

/* ── Split bar (80/10/10) ── */
.split{ display:flex; height:54px; border-radius:12px; overflow:hidden; margin:10px 0 6px;
  border:1px solid var(--border); }
.split div{ display:flex; align-items:center; justify-content:center; color:#fff;
  font-weight:700; font-size:.95rem; text-align:center; padding:0 6px; line-height:1.2; }

/* ── FAQ ── */
.qa{ background:#fff; border:1px solid var(--border); border-radius:14px; margin:0 0 12px;
  overflow:hidden; }
.qa-q{ padding:13px 18px; font-weight:700; font-size:1.03rem; color:var(--navy);
  background:#F7F9FC; }
.qa-q span{ background:var(--gold); color:var(--navy); border-radius:6px; padding:1px 8px;
  font-size:.8rem; margin-right:8px; }
.qa-a{ padding:13px 18px; font-size:1.02rem; line-height:1.7; color:#1E293B; }

/* ── Sidebar blocks ── */
.brand{ background:var(--navy); border-radius:16px; padding:18px 18px 16px; margin:4px 0 18px; }
.brand-pill{ display:inline-block; background:var(--gold); color:var(--navy); font-weight:800;
  font-size:.72rem; letter-spacing:.12em; padding:3px 10px; border-radius:999px; }
.brand-title{ color:#fff; font-size:1.35rem; font-weight:800; margin:10px 0 4px; }
.brand-course{ color:#CBD5E1; font-size:.9rem; }
.brand-inst{ color:var(--gold); font-size:.88rem; font-weight:600; margin-top:6px; }
.side-label{ font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
  color:#64748B; margin:18px 0 8px; }
.prog-text{ font-size:.9rem; font-weight:600; color:var(--navy); margin-bottom:6px; }
.prog{ height:8px; background:#E2E8F0; border-radius:99px; overflow:hidden; }
.prog div{ height:100%; background:var(--gold); border-radius:99px; }
.legend-item{ display:flex; align-items:center; gap:10px; font-size:.9rem; color:#1E293B;
  margin:6px 0; }
.legend-item i{ width:14px; height:14px; border-radius:4px; background:var(--c);
  display:inline-block; }
.setup-cmd{ display:block; font-family:'JetBrains Mono', monospace; font-size:.78rem;
  color:var(--navy); background:#F3F6FB; border:1px solid var(--border); border-radius:8px;
  padding:6px 10px; margin-top:6px; }

/* -- Big highlighted maths line inside a step -- */
.mbox{ display:block; font-family:'JetBrains Mono', monospace; font-size:1.34rem;
  font-weight:700; color:var(--navy); background:#F1F5F9;
  border-left:6px solid var(--c); border-radius:10px; padding:12px 17px;
  margin:11px 0 4px; letter-spacing:.005em; line-height:1.4; }
.mbox .dim{ color:#7A8BA0; font-weight:600; }
.mbox .hi{ color:var(--c); }

/* -- Before / after the projection: the key idea of the section -- */
.ba{ display:grid; grid-template-columns:1fr auto 1fr; gap:14px; align-items:stretch;
  margin:10px 0 24px; }
@media (max-width:900px){ .ba{ grid-template-columns:1fr; } }
.ba-box{ border:3px solid var(--c); border-radius:18px; padding:20px 22px;
  background:color-mix(in srgb, var(--c) 7%, white); }
.ba-tag{ font-size:.78rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase;
  color:var(--c); }
.ba-main{ font-size:1.42rem; font-weight:800; color:var(--navy); margin:8px 0 10px;
  line-height:1.25; }
.ba-code{ font-family:'JetBrains Mono', monospace; font-size:1.02rem; color:#475569;
  background:#fff; border-radius:8px; padding:8px 12px; display:block; }
.ba-mid{ display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:6px; }
.ba-arrow{ font-size:2.4rem; font-weight:800; color:var(--navy); line-height:1; }
.ba-note{ font-size:.86rem; font-weight:800; color:var(--muted); text-align:center;
  text-transform:uppercase; letter-spacing:.09em; max-width:120px; line-height:1.35; }

/* -- Filter -> number list -- */
.flist{ font-family:'JetBrains Mono', monospace; font-size:1.06rem; line-height:2;
  color:var(--navy); background:#F7F9FC; border-radius:11px; padding:13px 18px;
  margin:10px 0 2px; }
.flist b{ color:var(--purple); }

/* -- Segmented-control look for horizontal radios in the main area -- */
[data-testid="stMainBlockContainer"] [role="radiogroup"]{ gap:8px; flex-wrap:wrap; }
[data-testid="stMainBlockContainer"] [role="radiogroup"] label{
  background:#fff; border:2px solid var(--border); border-radius:11px;
  padding:9px 18px; margin:0; cursor:pointer; transition:border-color .12s ease; }
[data-testid="stMainBlockContainer"] [role="radiogroup"] label:hover{
  border-color:var(--blue); }
[data-testid="stMainBlockContainer"] [role="radiogroup"] label:has(input:checked){
  background:var(--navy); border-color:var(--navy); }
[data-testid="stMainBlockContainer"] [role="radiogroup"] label p{
  font-family:'JetBrains Mono', monospace !important; font-weight:700 !important;
  font-size:1.05rem !important; color:var(--navy) !important; }
[data-testid="stMainBlockContainer"] [role="radiogroup"] label:has(input:checked) p{
  color:#fff !important; }
[data-testid="stMainBlockContainer"] [role="radiogroup"] label > div:first-child{
  display:none; }

/* -- Dashboard stat strip (Splunk / Grafana style) -- */
.dash{ background:linear-gradient(145deg,#002147 0%,#01122A 100%); border-radius:20px;
  margin:10px 0 32px; display:flex; flex-wrap:wrap;
  box-shadow:0 12px 32px rgba(0,33,71,.26); overflow:hidden; }
.dtile{ flex:1 1 155px; padding:22px 16px 19px; text-align:center; position:relative; }
.dtile + .dtile::before{ content:""; position:absolute; left:0; top:20%; height:60%;
  width:1px; background:rgba(255,255,255,.15); }
.dval{ display:block; font-family:JetBrains Mono, monospace; font-size:3rem;
  font-weight:700; color:var(--c); line-height:1; letter-spacing:-.04em;
  text-shadow:0 0 26px color-mix(in srgb, var(--c) 50%, transparent); }
.dlbl{ display:block; font-size:.79rem; font-weight:800; color:#FFFFFF;
  text-transform:uppercase; letter-spacing:.15em; margin-top:13px; }
.ddesc{ display:block; font-size:.84rem; color:#8FA6C6; line-height:1.4; margin-top:7px; }
@media (max-width:900px){ .dval{ font-size:2.2rem; } .dtile{ flex:1 1 45%; } }

/* -- Status summary chips above the comparison table -- */
.sumrow{ display:flex; gap:10px; flex-wrap:wrap; margin:4px 0 14px; }
.sumchip{ display:flex; align-items:center; gap:9px; background:#fff;
  border:2px solid var(--c); border-radius:999px; padding:7px 16px 7px 13px; }
.sumchip .n{ font-family:JetBrains Mono, monospace; font-weight:700; font-size:1.3rem;
  color:var(--c); line-height:1; }
.sumchip .t{ font-size:.86rem; font-weight:800; color:var(--navy);
  text-transform:uppercase; letter-spacing:.08em; }

/* -- NLP vs ViT comparison table -- */
.vs{ background:#fff; border:1px solid var(--border); border-radius:18px; overflow:hidden;
  overflow-x:auto; margin:8px 0 26px; box-shadow:0 4px 16px rgba(0,33,71,.07); }
.vs table{ width:100%; border-collapse:collapse; min-width:680px; }
.vs th{ background:var(--navy); font-weight:800; text-align:left; padding:15px 18px;
  font-size:.95rem; letter-spacing:.06em; text-transform:uppercase; color:#fff; }
.vs th.nlpcol{ color:#9FC0EE; }
.vs th.vitcol{ color:var(--gold); }
.vs td{ padding:16px 18px; border-top:1px solid #E9EEF6; font-size:1.09rem;
  line-height:1.5; vertical-align:middle; }
.vs td.comp{ font-weight:800; color:var(--navy); white-space:nowrap;
  border-left:6px solid transparent; }
.vs tr.r-changed td.comp{ border-left-color:var(--red); }
.vs tr.r-same    td.comp{ border-left-color:var(--orange); }
.vs tr.r-ident   td.comp{ border-left-color:var(--green); }
.vs td.nlpcol{ color:#6B7A8D; }
.vs td.vitcol{ color:var(--navy); font-weight:600; }
.vs tr.r-changed td{ background:#FFF6F8; }
.vs tr.r-same    td{ background:#FFFCF0; }
.vs tr.r-ident   td{ background:#F2FDF7; }
.vs .vpill{ display:inline-flex; align-items:center; gap:7px; font-size:.78rem;
  font-weight:800; padding:6px 14px; border-radius:999px; text-transform:uppercase;
  letter-spacing:.08em; white-space:nowrap; color:#fff; }
.vs .vpill b{ font-size:.98rem; line-height:1; }

/* -- Stat cards (overview) -- */
.mrow{ display:flex; gap:14px; flex-wrap:wrap; margin:8px 0 30px; }
.mcard{ flex:1 1 165px; background:color-mix(in srgb, var(--c) 8%, white);
  border:2px solid var(--c); border-radius:18px; padding:20px 16px 17px; text-align:center;
  box-shadow:0 4px 14px rgba(0,33,71,.07); }
.mval{ display:block; font-family:'JetBrains Mono', monospace; font-size:2.6rem;
  font-weight:700; color:var(--c); line-height:1; letter-spacing:-.03em; }
.mlbl{ display:block; font-size:.97rem; font-weight:800; color:var(--navy);
  text-transform:uppercase; letter-spacing:.09em; margin-top:10px; }
.mdesc{ display:block; font-size:.93rem; color:var(--muted); line-height:1.45; margin-top:7px; }
@media (max-width:900px){ .mval{ font-size:2.1rem; } }

/* -- Patch / position pills -- */
.tok.patch{ background:#DBEAFE; color:var(--blue); border-color:#93C5FD; }
.tok.pos  { background:#FEF3C7; color:var(--orange); border-color:#FCD34D; }

/* -- Tensor shape chips -- */
.shapes{ display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin:8px 0 18px; }
.shape{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1.04rem;
  padding:10px 16px; border-radius:11px; border:2px solid var(--c); color:var(--c);
  background:color-mix(in srgb, var(--c) 9%, white); white-space:nowrap; }
.shape-op{ font-size:1.55rem; font-weight:800; color:var(--navy); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── HTML helpers ──────────────────────────────────────────────────────────────
def html(s):
    """Render raw HTML. Leading spaces and blank lines are removed so the
    markdown parser never turns indented HTML into a code block."""
    clean = "\n".join(line.strip() for line in s.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)

def section_header(num, total, title, subtitle):
    html(f"""
    <div class="sec-hdr">
      <span class="sec-pill">Section {num} of {total}</span>
      <div class="sec-title">{title}</div>
      <div class="sec-sub">{subtitle}</div>
      <div class="sec-rule"></div>
    </div>""")

def key_idea(text, hero=False):
    html(f'<div class="key{" hero" if hero else ""}">'
         f'<div class="key-label">Main idea</div>'
         f'<div class="key-text">{text}</div></div>')

def lead(text):
    html(f'<p class="lead">{text}</p>')

def hint(text):
    html(f'<p class="hint">{text}</p>')

def sub(text):
    html(f'<div class="sub">{text}</div>')

def tip(text):
    html(f'<div class="tip"><span class="tip-label">Class tip</span>{text}</div>')

def tag(text, color):
    return f'<span class="tag" style="--c:{color};">{text}</span>'

def card(title, body, color):
    return (f'<div class="card" style="--c:{color};">'
            f'<div class="card-title">{title}</div>'
            f'<div class="card-body">{body}</div></div>')

def grid(cards, cols=2):
    html(f'<div class="grid g{cols}">{"".join(cards)}</div>')

def steps(items):
    rows = "".join(
        f'<div class="step"><div class="num">{i}</div><div class="step-text">{t}</div></div>'
        for i, t in enumerate(items, 1))
    html(f'<div class="steps">{rows}</div>')

def table(headers, rows, num_cols=(), highlight_rows=()):
    """headers: list of str. rows: list of lists (cells may contain HTML)."""
    th = "".join(f'<th class="{"ncol" if i in num_cols else ""}">{h}</th>'
                 for i, h in enumerate(headers))
    body = ""
    for r, row in enumerate(rows):
        tds = "".join(f'<td class="{"ncol" if i in num_cols else ""}">{c}</td>'
                      for i, c in enumerate(row))
        body += f'<tr class="{"hl" if r in highlight_rows else ""}">{tds}</tr>'
    html(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{th}</tr></thead>'
         f'<tbody>{body}</tbody></table></div>')

def tok(text, kind):
    return f'<span class="tok {kind}">{text}</span>'

def keyrows(items):
    """Bulleted rows with a bold, colored lead-in term.
    items: list of (color, term, text). Pass term="" for a plain bullet."""
    lis = ""
    for color, term, text in items:
        lead_in = f'<span class="k">{term}</span> ' if term else ""
        lis += f'<li style="--c:{color};">{lead_in}{text}</li>'
    html(f'<ul class="blist">{lis}</ul>')

def faq(items):
    blocks = "".join(
        f'<div class="qa"><div class="qa-q"><span>Q{i}</span>{q}</div>'
        f'<div class="qa-a">{a}</div></div>' for i, (q, a) in enumerate(items, 1))
    html(blocks)


# ── Chart style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": "#D9E1EC",
    "axes.labelcolor": MUTED,
    "axes.titleweight": "bold",
    "axes.titlecolor": NAVY,
    "axes.titlesize": 13,
    "xtick.color": MUTED,
    "ytick.color": "#1E293B",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)




def vs_table(rows):
    """NLP vs ViT comparison. rows: list of (component, nlp, vit, status)
    where status is 'changed', 'same' or 'ident'."""
    pill = {"changed": ("Changed", RED, "&#10007;"),
            "same": ("Same idea", ORANGE, "&#8776;"),
            "ident": ("Identical", GREEN, "&#10003;")}
    order = ["ident", "same", "changed"]
    counts = {k: sum(1 for r in rows if r[3] == k) for k in order}
    chips = "".join(
        f'<div class="sumchip" style="--c:{pill[k][1]};">'
        f'<span class="n">{counts[k]}</span><span class="t">{pill[k][0]}</span></div>'
        for k in order if counts[k])
    html(f'<div class="sumrow">{chips}</div>')

    body = ""
    for comp, nlp, vit, status in rows:
        label, colour, glyph = pill[status]
        body += (f'<tr class="r-{status}"><td class="comp">{comp}</td>'
                 f'<td class="nlpcol">{nlp}</td><td class="vitcol">{vit}</td>'
                 f'<td><span class="vpill" style="background:{colour};">'
                 f'<b>{glyph}</b>{label}</span></td></tr>')
    html(f'<div class="vs"><table><thead><tr><th>Component</th>'
         f'<th class="nlpcol">Transformer (NLP)</th><th class="vitcol">ViT (vision)</th>'
         f'<th>Status</th></tr></thead><tbody>{body}</tbody></table></div>')

def pipeline(items, start=1):
    """Numbered vertical flow. items: list of (colour, title, body_html)."""
    out = ""
    for i, (c, t, b) in enumerate(items, start):
        if i > start:
            out += '<div class="pipe-link"><span>&#8595;</span></div>'
        out += (f'<div class="pipe-row"><div class="pipe-num" style="--c:{c};">{i}</div>'
                f'<div class="pipe-body" style="--c:{c};">'
                f'<div class="pipe-what">{t}</div>'
                f'<div class="pipe-why">{b}</div></div></div>')
    html(f'<div class="pipe">{out}</div>')

def dashboard(items):
    """Dark stat strip. items: (value, label, description, colour).
    Colours must read against navy, so use bright accents."""
    tiles = "".join(
        f'<div class="dtile" style="--c:{c};"><span class="dval">{v}</span>'
        f'<span class="dlbl">{l}</span><span class="ddesc">{d}</span></div>'
        for v, l, d, c in items)
    html(f'<div class="dash">{tiles}</div>')

def metrics(items):
    """items: list of (value, label, description, color)."""
    cards = "".join(
        f'<div class="mcard" style="--c:{c};"><span class="mval">{v}</span>'
        f'<span class="mlbl">{l}</span><span class="mdesc">{d}</span></div>'
        for v, l, d, c in items)
    html(f'<div class="mrow">{cards}</div>')

def shapes(items, op="&#8594;"):
    """Tensor-shape chips joined by an operator. items: list of (text, color)."""
    chips = f'<span class="shape-op">{op}</span>'.join(
        f'<span class="shape" style="--c:{c};">{t}</span>' for t, c in items)
    html(f'<div class="shapes">{chips}</div>')

def patch_row(labels):
    """labels: list of (text, kind) where kind is cls / patch / pos / sep / out."""
    pills = "".join(tok(t, k) for t, k in labels)
    html(f'<div class="tok-row">{pills}</div>')


# -- Image handling ----------------------------------------------------------
# google/vit-* was trained with mean = std = 0.5 (NOT the ImageNet statistics),
# so we use its own convention everywhere. That keeps the pretrained demos
# correct, and for the untrained layers the choice makes no difference.
VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD  = [0.5, 0.5, 0.5]

def preprocess_image(img, size=224):
    """PIL image -> normalized (1, 3, size, size) tensor. No torchvision needed."""
    arr = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)                       # (3, H, W)
    mean = torch.tensor(VIT_MEAN).view(3, 1, 1)
    std  = torch.tensor(VIT_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)                           # (1, 3, H, W)

def upscale(grid, size=224):
    """Bilinearly resize a 2-D numpy map up to size x size, using torch.

    Replaces cv2.resize so the app needs no OpenCV install."""
    t = torch.from_numpy(np.asarray(grid, dtype=np.float32))[None, None]
    up = torch.nn.functional.interpolate(t, size=(size, size),
                                         mode="bilinear", align_corners=False)
    return up[0, 0].numpy()

def sinusoidal_pe(n_pos, d):
    """The ORIGINAL Transformer position encoding: a fixed sin/cos formula with
    zero learnable parameters. Used here only for contrast; ViT does not use it."""
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d)
    pe = np.zeros((n_pos, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

def centre_similarity(pe, side=14, row=6, col=6):
    """Cosine similarity of one patch position to the whole grid, as (side, side)."""
    pe = np.asarray(pe, dtype=np.float32)
    pe = pe / (np.linalg.norm(pe, axis=-1, keepdims=True) + 1e-8)
    sim = pe @ pe.T
    return sim[1 + row * side + col, 1:].reshape(side, side)

def normalize01(a):
    a = np.asarray(a, dtype=np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


@st.cache_resource(show_spinner="Loading pretrained ViT-Base/16 (about 30 s the first time)...")
def load_vit():
    from transformers import ViTModel
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k",
                                     output_attentions=True)
    model.eval()
    return model

def try_load_vit():
    """load_vit(), but returns None and explains itself instead of showing a
    traceback. The checkpoint is about 330 MB, which a small hosted container
    can fail to download or hold in memory."""
    try:
        return load_vit()
    except Exception as exc:
        st.error(
            "Could not load the pretrained ViT-Base checkpoint "
            "(about 330 MB).\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "Everything else in this app runs without it. This section needs the "
            "real trained weights, so it is the one place that depends on the "
            "download succeeding. On a small hosted container this is usually "
            "memory or disk, not your code: running locally will work."
        )
        return None


# Parameter count of the ViT-Base built in Section 8 of this app. Published
# figures for ViT-Base/16 sit a little under this (around 86.4M); the gap is
# bias terms and head size, not a different architecture.
VIT_BASE_TOTAL = 86_540_008


# -- Navigation --------------------------------------------------------------
SECTIONS = [
    "The Bridge: Words to Patches",
    "Patch Extraction",
    "Patch Embedding",
    "CLS Token and Position",
    "Self-Attention over Patches",
    "The Encoder Block",
    "The Classification Head",
    "The Full Model",
    "Training in Practice",
]
LABELS = [f"{i}.  {s}" for i, s in enumerate(SECTIONS, 1)]
TOTAL = len(SECTIONS)

if "sec" not in st.session_state:
    st.session_state.sec = LABELS[0]

def go(delta):
    i = LABELS.index(st.session_state.sec)
    st.session_state.sec = LABELS[max(0, min(TOTAL - 1, i + delta))]

def default_image():
    """A smooth synthetic image, used until the student uploads one."""
    i, j = np.meshgrid(np.arange(224), np.arange(224), indexing="ij")
    arr = np.stack([
        (255 * (0.5 + 0.5 * np.sin(i / 30))).astype(np.uint8),
        (255 * (0.5 + 0.5 * np.cos(j / 25))).astype(np.uint8),
        (255 * (0.5 + 0.5 * np.sin((i + j) / 40))).astype(np.uint8),
    ], axis=-1)
    return Image.fromarray(arr)

with st.sidebar:
    html("""
    <div class="brand">
      <span class="brand-pill">SEAS 8525</span>
      <div class="brand-title">ViT Walkthrough</div>
      <div class="brand-course">Computer Vision &amp; Generative AI</div>
      <div class="brand-inst">Dr. Elbasheer</div>
    </div>""")

    st.radio("Sections", LABELS, key="sec", label_visibility="collapsed")
    idx = LABELS.index(st.session_state.sec)

    html(f"""
    <div class="side-label">Progress</div>
    <div class="prog-text">Section {idx + 1} of {TOTAL}</div>
    <div class="prog"><div style="width:{(idx + 1) / TOTAL * 100:.0f}%;"></div></div>
    <div class="side-label">Your image</div>""")

    uploaded = st.file_uploader("Used by every live demo",
                               type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        user_img = Image.open(uploaded).convert("RGB")
        st.image(user_img, caption="Your image", use_container_width=True)
    else:
        user_img = default_image()
        st.image(user_img, caption="Default image (upload your own)",
                 use_container_width=True)

    html("""
    <div class="side-label">Quick setup</div>
    <span class="setup-cmd">pip install -r requirements.txt</span>
    <span class="setup-cmd">streamlit run vit_app.py</span>
    """)

num = idx + 1
img224 = user_img.resize((224, 224))



# ==============================================================================
# SECTION 1: THE BRIDGE
# ==============================================================================
if num == 1:
    section_header(1, TOTAL, "The Bridge: Words to Patches",
                   "You already know the encoder. ViT changes only what goes into it.")

    key_idea("ViT invents <b>no new architecture</b>. It asks one question: what if an "
             "image were a <b>sequence of 16x16 patches</b> instead of a sentence of "
             "words? The transformer encoder is then reused <b>completely unchanged</b>.",
             hero=True)

    dashboard([
        ("86M", "Parameters", "learnable weights", GOLD),
        ("12",  "Layers",     "encoder blocks stacked", "#7DD3FC"),
        ("196", "Patches",    "tokens per image", "#6EE7B7"),
        ("768", "Embed dim",  "numbers per token", "#C4B5FD"),
        ("12",  "Heads",      "per attention layer", "#FDBA74"),
    ])

    sub("The whole model in one line")
    shapes([("image (3, 224, 224)", BLUE), ("196 patches", BLUE),
            ("(197, 768) + CLS", PURPLE), ("encoder x12", ORANGE),
            ("class label", GREEN)])

    sub("What changes, what stays identical")
    vs_table([
        ("Input unit",        "Word token",              "16x16 pixel patch",      "changed"),
        ("Tokenizing",        "Vocabulary lookup",       "Flatten + project",      "changed"),
        ("Sequence length",   "Number of words",         "196 patches + CLS",      "changed"),
        ("Position encoding", "Sinusoidal or learned",   "Learned, one per slot",  "same"),
        ("Self-attention",    "Multi-head Q/K/V",        "Multi-head Q/K/V",       "ident"),
        ("Feed-forward",      "2-layer MLP",             "2-layer MLP",            "ident"),
        ("Add and Norm",      "Residual + LayerNorm",    "Residual + LayerNorm",   "ident"),
        ("Output head",       "Linear over last token",  "Linear over CLS",        "changed"),
    ])

    sub("Two questions students always ask")
    keyrows([
        (BLUE, "Why patches, not pixels?",
         "Attention cost is <b>O(N squared)</b>."),
        (BLUE, "The numbers.",
         "Pixels: N = 50,176, about <b>2.5 billion</b> pairs. Patches: N = 196, "
         "<b>38,416</b> pairs. Roughly <b>65,000x cheaper</b>."),
        (ORANGE, "Why ViT needs so much data.",
         "A CNN is born knowing <b>locality</b> and <b>translation equivariance</b>. "
         "ViT has <b>neither</b>."),
        (ORANGE, "So:",
         "small data, CNNs win. Pretrained on <b>ImageNet-21k</b> or <b>JFT-300M</b>, "
         "ViT wins."),
    ])

    with st.expander("Practice questions"):
        faq([
            ("What inductive biases do CNNs have that ViT lacks?",
             "Locality and translation equivariance. ViT treats every pair of patches "
             "alike, so it must learn spatial structure from data."),
            ("Why does ViT beat CNNs only on large datasets?",
             "Those biases are free generalisation on small data. ViT has to learn them, "
             "which takes scale, but it can then exploit global context convolutions miss."),
            ("What replaces the word-embedding lookup table?",
             "A single learned linear projection, implemented as "
             "<code>Conv2d(kernel=16, stride=16)</code>. Section 3 builds it."),
            ("What is the CLS token for?",
             "A learnable vector prepended to the patches. Over 12 layers it collects "
             "information from all of them, and only its final output reaches the "
             "classifier."),
            ("Why is self-attention's cost a problem for images?",
             "It is quadratic in sequence length. Patches shrink the sequence from 50,176 "
             "to 196, about 65,000 times fewer attention pairs."),
            ("How does ViT's position encoding differ?",
             "It is <b>learned</b>, one vector per patch position, rather than fixed "
             "sinusoids. Despite being 1-D it develops 2-D spatial structure on its own."),
            ("What does attention rollout show?",
             "Attention multiplied through all 12 layers, revealing which patches most "
             "shaped the CLS vector. It often highlights the object with no segmentation "
             "labels at all."),
        ])


# ==============================================================================
# SECTION 2: PATCH EXTRACTION
# ==============================================================================
elif num == 2:
    section_header(2, TOTAL, "Patch Extraction",
                   "Cutting the image into a sequence of fixed-size tiles")

    key_idea("Split an <b>H x W</b> image into non-overlapping <b>P x P</b> tiles. That "
             "gives <b>N = (H/P) x (W/P)</b> patches, and each one is flattened into a "
             "single vector. The image has become a sequence.")

    shapes([("(3, 224, 224)", BLUE), ("14 x 14 grid", PURPLE),
            ("(196, 768)", GREEN)])

    keyrows([
        (BLUE, "The arithmetic, for ViT-Base.",
         "224 / 16 = 14, so a <b>14 x 14 grid</b> = <b>196 patches</b>. The demo below "
         "lets you change both numbers."),
        (PURPLE, "Each patch.",
         "16 x 16 x 3 = <b>768 raw numbers</b>. Matching ViT's embedding size here is "
         "a coincidence; the two are set independently."),
        (ORANGE, "No overlap.",
         "Patches tile the image edge to edge. Nothing counted twice, nothing skipped."),
    ])

    sub("Try it: choose the image size and the patch size")
    hint("Patch size is the <b>biggest single lever</b> on cost. Smaller patches mean "
         "finer detail and a far longer, far more expensive sequence.")

    PATCH_CHOICES = [4, 7, 8, 14, 16, 28, 32, 56, 64]

    c1, c2 = st.columns([1, 1.5])
    with c1:
        img_size = st.number_input("Original image size (square, pixels)",
                                   min_value=64, max_value=512, value=224, step=8,
                                   help="ViT-Base uses 224. Patch size must divide this "
                                        "evenly, so the options on the right update.")
    with c2:
        valid = [q for q in PATCH_CHOICES if img_size % q == 0]
        default = 16 if 16 in valid else valid[len(valid) // 2]
        patch_size = st.radio(
            f"Patch size P  ({len(valid)} options divide {img_size} evenly)",
            valid, index=valid.index(default), horizontal=True,
            format_func=lambda v: f"{v}px")

    side  = img_size // patch_size
    n_pat = side ** 2
    vec   = patch_size * patch_size * 3
    pairs = (n_pat + 1) ** 2

    dashboard([
        (f"{img_size}", "Image size", "pixels, square", "#C4B5FD"),
        (f"{side}x{side}", "Patch grid", "across and down", GOLD),
        (f"{n_pat}", "Patches", "sequence length N", "#7DD3FC"),
        (f"{vec}", "Flatten vector", f"{patch_size} x {patch_size} x 3 numbers", "#6EE7B7"),
        (f"{pairs:,}", "Attention pairs", "with CLS, the real cost", "#FDBA74"),
    ])

    img_n = user_img.resize((img_size, img_size))

    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.image(img_n, caption=f"Input, {img_size} x {img_size}",
                 use_container_width=True)
    with c2:
        keyrows([
            (BLUE, "Into the encoder:",
             f"<b>({n_pat + 1}, 768)</b>, the {n_pat} patches plus CLS."),
            (ORANGE, "Halve the patch size,",
             "and the sequence <b>quadruples</b>. Attention cost rises about <b>16x</b>."),
            (PURPLE, "Why P must divide the image.",
             "Patches tile it exactly, so any leftover strip would have nowhere to go."),
        ])
        run = st.button("Extract patches", type="primary")

    if run:
        with st.spinner("Extracting..."):
            img_tensor = preprocess_image(img_n, size=img_size)
            B, C, H, W = img_tensor.shape
            P = patch_size
            # unfold tiles the image without overlap, then we flatten each tile
            patches = (img_tensor.unfold(2, P, P).unfold(3, P, P)
                       .contiguous().view(B, C, -1, P * P)
                       .permute(0, 2, 3, 1).flatten(2))

        dashboard([
            (str(tuple(patches.shape)), "patches.shape", "batch, N, flat vector", GOLD),
            (f"{n_pat}", "Patches", "one token each", "#7DD3FC"),
            (f"{vec}", "Flatten vector", "numbers per patch", "#6EE7B7"),
        ])

        arr = np.array(img_n)
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
        axes[0].imshow(arr)
        axes[0].set_title("Original", loc="left")
        axes[1].imshow(arr)
        lw = 1.0 if side <= 32 else 0.4
        for i in range(side + 1):
            axes[1].axhline(i * P, color=GOLD, lw=lw)
            axes[1].axvline(i * P, color=GOLD, lw=lw)
        axes[1].set_title(f"{side} x {side} grid of {P}x{P} patches", loc="left")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        show = min(12, n_pat)
        sub(f"The first {show} patches, in the order the encoder reads them")
        cols = 6
        rows = (show + cols - 1) // cols
        fig2, axes2 = plt.subplots(rows, cols, figsize=(11, 2.0 * rows), squeeze=False)
        for k in range(rows * cols):
            ax = axes2[k // cols][k % cols]
            ax.axis("off")
            if k < show:
                r, c = k // side, k % side
                ax.imshow(arr[r * P:(r + 1) * P, c * P:(c + 1) * P])
                ax.set_title(f"patch {k}", fontsize=9, loc="left")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        hint(f"Each one is now a flat vector of <b>{vec}</b> numbers. The encoder never "
             "sees the grid, only the sequence, which is exactly why position embeddings "
             "are needed in Section 4.")

    with st.expander("Show the patch-extraction code"):
        st.code('''
import torch

def extract_patches(x, patch_size=16):
    """(B, 3, 224, 224) -> (B, 196, 768). Pure PyTorch, no extra libraries."""
    B, C, H, W = x.shape
    P = patch_size
    # unfold twice: once down the height, once across the width
    x = x.unfold(2, P, P).unfold(3, P, P)   # (B, C, 14, 14, P, P)
    x = x.contiguous().view(B, C, -1, P * P)  # (B, C, 196, 256)
    x = x.permute(0, 2, 3, 1).flatten(2)      # (B, 196, 768)
    return x

# In practice ViT never does this explicitly: the Conv2d in Section 3
# performs the tiling AND the projection in a single operation.
''', language="python")


# ==============================================================================
# SECTION 3: PATCH EMBEDDING
# ==============================================================================
elif num == 3:
    section_header(3, TOTAL, "Patch Embedding",
                   "Two ways to turn a patch into a vector: by hand, or with a Conv2d")

    key_idea("A patch is still just <b>raw pixels</b>. ViT turns it into a <b>learned "
             "representation</b> first. There are <b>two ways</b> to do that: "
             "<b>by hand</b>, or with a <b>Conv2d</b>. Both give the same answer.",
             hero=True)

    dashboard([
        ("2", "Routes", "manual, or Conv2d", "#7DD3FC"),
        ("196", "Patches", "one embedding each", "#6EE7B7"),
        ("768", "Raw numbers", "16 x 16 x 3 per patch", "#FDBA74"),
        ("589,824", "Shared weights", "768 x 768, one matrix", GOLD),
    ])

    # ============================ shared starting point ======================
    sub("What both routes start from")
    pipeline([
        (BLUE, "The image",
         '<span class="mbox" style="--c:#1D4ED8;">224 x 224 x 3</span>'),
        (BLUE, "Cut into patches",
         '<span class="mbox" style="--c:#1D4ED8;">224 / 16 = <span class="hi">14</span>'
         '&nbsp;&nbsp;so&nbsp;&nbsp;14 x 14 = <span class="hi">196 patches</span></span>'),
        (ORANGE, "One patch, flattened",
         '<span class="mbox" style="--c:#D97706;">16 x 16 x 3 = '
         '<span class="hi">768</span> raw pixel values</span>'
         '<span class="mbox" style="--c:#D97706;">[ 12, 34, 55, ... ]'
         '<span class="dim">&nbsp;&nbsp;brightness, nothing learned yet</span></span>'),
    ])

    # ============================ the projection idea ========================
    sub("What the projection changes")
    lead("Both routes multiply that flat patch by <b>one learned matrix</b>, the same "
         "matrix for all 196 patches.")
    html('<span class="mbox" style="--c:#7C3AED;">W &#8712; R<sup>768 x 768</sup>'
         '&nbsp;&nbsp;so&nbsp;&nbsp;768 <span class="hi">&#8594;</span> 768</span>')

    html(f"""
    <div class="ba">
      <div class="ba-box" style="--c:{ORANGE};">
        <div class="ba-tag">Before the projection</div>
        <div class="ba-main">Raw RGB pixel values</div>
        <span class="ba-code">[ 12, 34, 55, ... ]</span>
      </div>
      <div class="ba-mid">
        <div class="ba-arrow">&#8594;</div>
        <div class="ba-note">same 768 numbers</div>
      </div>
      <div class="ba-box" style="--c:{GREEN};">
        <div class="ba-tag">After the projection</div>
        <div class="ba-main">Learned visual features</div>
        <span class="ba-code">[ z&#8321;, z&#8322;, ... z&#8327;&#8326;&#8328; ]</span>
      </div>
    </div>""")

    keyrows([
        (RED, "The size did not change. So what did?",
         "<b>The meaning of the numbers.</b> Same length, completely different content."),
        (GREEN, "Exactly like word embeddings.",
         "A word is already a number, its vocabulary ID. The embedding layer learns a "
         "representation that is <b>useful to the model</b>. Same idea, with pixels."),
    ])

    # ============================ ROUTE 1 ====================================
    sub("Route 1: do it by hand")
    keyrows([
        (BLUE, "Perfectly correct.",
         "Three explicit steps. This is what the Conv2d is doing underneath."),
    ])
    pipeline([
        (BLUE, "Flatten all 196 patches",
         'Each becomes a row of 768 raw numbers.'
         '<span class="mbox" style="--c:#1D4ED8;">(196, 768)'
         '<span class="dim">&nbsp;&nbsp;raw pixels</span></span>'),
        (BLUE, "Multiply every row by the same W",
         'One matrix, reused for all 196 patches. Not 196 different matrices.'
         '<span class="mbox" style="--c:#1D4ED8;">(196, 768) x (768, 768) = '
         '<span class="hi">(196, 768)</span></span>'),
        (GREEN, "Stack the results",
         'You now have the sequence the encoder wants.'
         '<span class="mbox" style="--c:#0E9F6E;">(196, 768)'
         '<span class="dim">&nbsp;&nbsp;learned features</span></span>'),
    ])

    # ============================ ROUTE 2 ====================================
    sub("Route 2: let a Conv2d do it")
    keyrows([
        (GREEN, "Conv2d does both at once.",
         "One operation covers <b>extract the patch</b> and <b>project it</b>, for all "
         "196 patches, on the GPU."),
    ])
    pipeline([
        (GREEN, "Set kernel and stride to the patch size",
         '<span class="mbox" style="--c:#0E9F6E;">nn.Conv2d(3, 768, '
         'kernel_size=<span class="hi">16</span>, stride=<span class="hi">16</span>)</span>'
         'The kernel sits on the image, then jumps <b>exactly 16 pixels</b>. Because '
         '<b>stride equals kernel size</b> the patches never overlap, and there is '
         'one kernel position per patch: a <b>14 x 14</b> grid.'),
        (PURPLE, "Ask for 768 filters, not 64",
         'Each filter covers the <b>whole</b> patch, so it holds exactly as many weights '
         'as the patch has numbers.'
         '<span class="mbox" style="--c:#7C3AED;">one filter = 3 x 16 x 16 = '
         '<span class="hi">768</span> weights'
         '<span class="dim">&nbsp;&nbsp;= one row of W</span></span>'),
        (PURPLE, "Each filter returns one number",
         'Point all 768 filters at the <b>same</b> patch. Every filter reads the whole '
         'patch and answers with a single number.'
         '<span class="flist">one patch (3 x 16 x 16)<br>'
         '&nbsp;&nbsp;&#9500;&#9472; filter 1 &nbsp;&#8594; <b>z&#8321;</b><br>'
         '&nbsp;&nbsp;&#9500;&#9472; filter 2 &nbsp;&#8594; <b>z&#8322;</b><br>'
         '&nbsp;&nbsp;&#9500;&#9472; <span class="dim">&#8942;</span><br>'
         '&nbsp;&nbsp;&#9492;&#9472; filter 768 &#8594; <b>z&#8327;&#8326;&#8328;</b>'
         '</span>'
         'Those <b>768 answers, stacked, are that patch\u2019s embedding</b>. The same '
         'thing Route 1 computed with a matrix multiply.'),
        (NAVY, "Turn the grid into a sequence",
         'Conv2d hands back a <b>grid</b>, one position per patch. The encoder wants a '
         '<b>list of tokens</b>, so the 14 x 14 is flattened.'
         '<span class="mbox" style="--c:#002147;">[B, 768, 14, 14]'
         '<span class="dim">&nbsp;&nbsp;grid</span></span>'
         '<span class="mbox" style="--c:#002147;">[B, <span class="hi">196</span>, 768]'
         '<span class="dim">&nbsp;&nbsp;196 tokens, 768 features each</span></span>'),
    ])

    tip("So the convolution is <b>not</b> here for the usual CNN reason of sliding one "
        "pixel at a time to find edges. With <b>stride equal to kernel size</b> it is "
        "simply a fast way to do <b>extract patch + linear projection</b> in one call. "
        "Route 1 and Route 2 are the same function.")

    sub("The result, either way")
    html("""
    <div class="flist" style="font-size:1.1rem;">
      patch 1 &#8594; <b>768-dimensional embedding</b><br>
      patch 2 &#8594; <b>768-dimensional embedding</b><br>
      patch 3 &#8594; <b>768-dimensional embedding</b><br>
      <span class="dim">&#8942;</span><br>
      patch 196 &#8594; <b>768-dimensional embedding</b>
    </div>""")

    # ============================ demo =======================================
    sub("Try it: run the patch embedding")
    hint("The Conv2d route on your image, exactly as ViT does it.")

    class PatchEmbedding(nn.Module):
        def __init__(self, patch_size=16, embed_dim=768):
            super().__init__()
            self.projection = nn.Conv2d(3, embed_dim, patch_size, patch_size)

        def forward(self, x):
            return self.projection(x).flatten(2).transpose(1, 2)

    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.image(img224, caption="Input image", use_container_width=True)
    with c2:
        run = st.button("Run the patch embedding", type="primary")

    if run:
        with st.spinner("Running..."):
            layer = PatchEmbedding()
            with torch.no_grad():
                out = layer(preprocess_image(img224))

        dashboard([
            ("[1, 3, 224, 224]", "Went in", "the image", "#FDBA74"),
            (str(tuple(out.shape)), "Came out", "196 tokens, 768 each", "#6EE7B7"),
            ("196", "Patches", "sequence length", "#7DD3FC"),
            ("589,824", "Weights used", "one shared matrix", GOLD),
        ])

        sub("One line per patch")
        fig, ax = plt.subplots(figsize=(10, 3.2))
        cols = [BLUE, GREEN, ORANGE, PURPLE, RED, "#0284C7", "#7C3AED", "#0F766E"]
        for i in range(8):
            ax.plot(out[0, i, :64].numpy(), color=cols[i], lw=1.3, alpha=.85,
                    label=f"patch {i}")
        ax.legend(loc="upper right", fontsize=8, ncol=4, frameon=False)
        ax.set_xlabel("Embedding dimension (0 to 63 of 768)")
        ax.set_ylabel("Value")
        ax.set_title("Different patches give different feature vectors", loc="left")
        style_axes(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        sub("The filters themselves")
        w = layer.projection.weight
        fig2, axes = plt.subplots(2, 4, figsize=(8.5, 4.4))
        for k, ax in enumerate(axes.flatten()):
            f = w[k].detach().numpy().transpose(1, 2, 0)
            ax.imshow((f - f.min()) / (f.max() - f.min() + 1e-8))
            ax.set_title(f"filter {k}", fontsize=8, loc="left")
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        hint("Each filter is <b>3 x 16 x 16 = 768</b> weights, reshaped back into a 16x16 "
             "colour tile. <b>Noise</b> here, because this layer is untrained. Trained, "
             "they become <b>edge, colour and texture detectors</b>, much like a CNN "
             "first layer.")

    with st.expander("Show both routes in code"):
        st.code("""
import torch, torch.nn as nn

# ---------- Route 1: the manual way. Correct, and slow. ----------------------
def manual_patch_embed(x, W, b, P=16):
    \"\"\"x: (B, 3, 224, 224) -> (B, 196, 768)\"\"\"
    B = x.shape[0]
    flat = (x.unfold(2, P, P).unfold(3, P, P)     # (B, 3, 14, 14, 16, 16)
             .permute(0, 2, 3, 1, 4, 5)           # (B, 14, 14, 3, 16, 16)
             .reshape(B, -1, 3 * P * P))          # (B, 196, 768)
    return flat @ W.T + b                         # one shared 768 x 768 matrix

# ---------- Route 2: the Conv2d way. Identical maths, one call. --------------
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768):
        super().__init__()
        # kernel_size = stride = patch_size, so the kernel jumps exactly one
        # patch at a time and the patches never overlap. The 768 filters ARE
        # the 768 rows of the projection matrix.
        self.projection = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x):          # x: (B, 3, 224, 224)
        x = self.projection(x)     # (B, 768, 14, 14)   one number per filter per patch
        x = x.flatten(2)           # (B, 768, 196)
        return x.transpose(1, 2)   # (B, 196, 768)      the sequence the encoder wants

# ---------- They are the same function ---------------------------------------
layer = PatchEmbedding()
W = layer.projection.weight.view(768, -1)      # (768, 3*16*16) = (768, 768)
b = layer.projection.bias

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    print((layer(x) - manual_patch_embed(x, W, b)).abs().max())   # ~2e-06

# The ONLY subtlety: flatten the patch in (C, P, P) order, because that is how
# the Conv2d weight is laid out. Flatten pixel-major instead and the numbers
# will not match, even though both are "flattening the patch".
""", language="python")


# ==============================================================================
# SECTION 4: CLS TOKEN AND POSITION
# ==============================================================================
elif num == 4:
    section_header(4, TOTAL, "CLS Token and Position",
                   "Two additions that turn a bag of patches into an ordered sequence")

    key_idea("Prepend <b>one learnable CLS vector</b> to carry the summary, then add "
             "<b>one learned position vector</b> to every slot. The sequence goes from "
             "<b>(196, 768)</b> to <b>(197, 768)</b> and is ready for the encoder.")

    shapes([("patches (196, 768)", BLUE), ("+ CLS (197, 768)", RED),
            ("+ position", ORANGE), ("encoder input", GREEN)])

    patch_row([("[CLS]", "cls")] + [(f"patch {i}", "patch") for i in range(5)]
              + [("...", "patch"), ("patch 195", "patch")])

    sub("Why each one is needed")
    keyrows([
        (RED, "The CLS token.",
         "One learnable vector, identical for every image. It carries <b>no image "
         "content</b>, so it is free to fill up with whatever the patches contribute."),
        (RED, "Only CLS is read.",
         "Its final output alone reaches the classifier, which keeps the encoder "
         "identical to the NLP version."),
        (ORANGE, "Why position matters.",
         "Attention is <b>permutation invariant</b>. Without positions, a face and a "
         "scrambled face look the same."),
        (ORANGE, "What ViT adds.",
         "One learned 768-vector per slot. <b>197</b> of them."),
        (PURPLE, "Learned, and only 1-D.",
         "ViT is never told that patch 14 sits below patch 0. It gets a flat index and "
         "works the geometry out. <b>The demo below proves it.</b>"),
    ])

    sub("What kind of positional encoding is this?")
    lead("There are <b>three</b> families in common use. ViT picks the middle one.")
    table(
        ["Family", "How it is produced", "Used by", "In ViT?"],
        [
            ["Fixed sinusoidal",
             "Computed from a <b>sin/cos formula</b>. <b>Zero</b> learnable parameters, "
             "and it extends to any sequence length.",
             "The original Transformer (2017)", tag("No", RED)],
            ["Learned absolute",
             "A <b>lookup table</b>: one trained vector per position <i>index</i>. "
             "Learned by backpropagation like any other weight.",
             f"BERT, {tag('ViT', GREEN)}", tag("Yes", GREEN)],
            ["Relative",
             "Encodes the <b>distance between</b> two tokens, injected into the attention "
             "scores rather than added to the input.",
             "Swin, T5, RoPE", tag("No", RED)],
        ],
        highlight_rows=(1,),
    )

    sub("How it actually happens in ViT")
    keyrows([
        (ORANGE, "It is a plain parameter.",
         "<code>nn.Parameter(torch.zeros(1, 197, 768))</code>. One 768-vector per slot, "
         "including one for CLS at index 0."),
        (ORANGE, "It is added, not concatenated.",
         "<code>x = cat([cls, patches]) + pos_embed</code>. The width stays <b>768</b>, "
         "so the encoder sees no change in shape."),
        (PURPLE, "It is learned from scratch.",
         "Initialised <code>trunc_normal(std=0.02)</code>, then trained by backprop. "
         "<b>Nothing about the 2-D layout is hardcoded.</b>"),
        (BLUE, "It is absolute, and 1-D.",
         "Position <b>14</b> is just the index 14. ViT is never told it sits directly "
         "below position 0."),
        (GREEN, "Why not 2-D, or relative?",
         "The ViT paper tried both, and also a relative scheme. None gave a significant "
         "gain over plain 1-D learned, so the simplest option stayed."),
        (RED, "Change the image size and you must interpolate.",
         "Because it is absolute, the table is tied to the 14 x 14 grid. Fine-tuning at a "
         "different resolution <b>2-D interpolates</b> these vectors onto the new grid."),
    ])

    sub("Try it: build the encoder input, then compare the two encodings")
    hint("Two things at once. The <b>untrained</b> module shows the shapes, and the "
         "<b>pretrained</b> ViT shows what the position embeddings actually learned. "
         "First run downloads about 330 MB.")

    class ViTEmbedding(nn.Module):
        def __init__(self, num_patches=196, embed_dim=768, patch_size=16):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, 196, 768)
            cls = self.cls_token.expand(B, -1, -1)               # (B,   1, 768)
            return torch.cat([cls, x], dim=1) + self.pos_embed   # (B, 197, 768)

    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.image(img224, caption="Input image", use_container_width=True)
    with c2:
        run = st.button("Run the embedding step", type="primary")

    if run:
        with st.spinner("Running the embedding module..."):
            mod = ViTEmbedding()
            with torch.no_grad():
                patches_only = mod.patch_embed(
                    preprocess_image(img224)).flatten(2).transpose(1, 2)
                out = mod(preprocess_image(img224))

        dashboard([
            (str(tuple(patches_only.shape)), "Patches only", "before CLS", "#7DD3FC"),
            (str(tuple(out.shape)), "With CLS", "ready for the encoder", "#6EE7B7"),
            ("197", "Positions", "196 patches plus CLS", "#FDBA74"),
            (f"{(mod.cls_token.numel() + mod.pos_embed.numel()) / 1e3:.0f}K",
             "CLS + pos params", "learned, not fixed", GOLD),
        ])

        sub("What the position embeddings actually learned")
        with st.spinner("Loading pretrained ViT-Base/16..."):
            vit = try_load_vit()
        if vit is None:
            st.stop()
        pos = vit.embeddings.position_embeddings[0].detach()          # (197, 768)
        pn = pos / (pos.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (pn @ pn.T).numpy()

        centre = 1 + 6 * 14 + 6                  # patch at grid row 6, col 6
        grid = sim[centre, 1:].reshape(14, 14)

        sin_grid = centre_similarity(sinusoidal_pe(197, 768))

        cmap = LinearSegmentedColormap.from_list("gwu", ["#FFFFFF", "#93C5FD", BLUE, NAVY])
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

        idx_map = np.arange(196).reshape(14, 14)
        im0 = axes[0].imshow(idx_map, cmap="cividis", interpolation="nearest")
        axes[0].set_title("1. What ViT is told: a flat index", loc="left")
        for r, c, lab in [(0, 0, "0"), (0, 13, "13"), (13, 0, "182"), (13, 13, "195")]:
            axes[0].text(c, r, lab, ha="center", va="center", fontsize=7,
                         color="white", fontweight="bold")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(grid, cmap=cmap, interpolation="nearest")
        axes[1].set_title("2. ViT LEARNED: 2-D locality", loc="left")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(sin_grid, cmap=cmap, interpolation="nearest")
        axes[2].set_title("3. FIXED sinusoidal: rows only", loc="left")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        for ax in (axes[1], axes[2]):
            ax.plot(6, 6, marker="*", color=GOLD, markersize=19,
                    markeredgecolor=NAVY, markeredgewidth=1.4, zorder=5)
        for ax in axes:
            ax.set_xlabel("patch column")
            ax.set_ylabel("patch row")
            for sp in ax.spines.values():
                sp.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        rr, cc = np.meshgrid(np.arange(14), np.arange(14), indexing="ij")
        off = (rr != 6) & (cc != 6)

        def stats(g):
            return (g[6, 7], g[7, 6], g[6, :].mean(), g[:, 6].mean(), g[off].mean())

        lr, lb, lrow, lcol, loth = stats(grid)
        sr, sb, srow, scol, soth = stats(sin_grid)

        sub("The same measurement on both encodings")
        table(
            ["Similarity of the starred patch to", f"{tag('ViT learned', GREEN)}",
             f"{tag('Fixed sinusoidal', RED)}"],
            [["the patch to its <b>right</b>", f"<b>{lr:+.3f}</b>", f"<b>{sr:+.3f}</b>"],
             ["the patch <b>below</b> it", f"<b>{lb:+.3f}</b>", f"<b>{sb:+.3f}</b>"],
             ["everything in its row (mean)", f"{lrow:+.3f}", f"{srow:+.3f}"],
             ["everything in its column (mean)", f"{lcol:+.3f}", f"{scol:+.3f}"],
             ["everything off its row and column", f"{loth:+.3f}", f"{soth:+.3f}"]],
            num_cols=(1, 2), highlight_rows=(0, 1),
        )
        keyrows([
            (RED, "The fixed encoding only knows a line.",
             f"Right scores <b>{sr:+.2f}</b> but below only <b>{sb:+.2f}</b>, because in "
             f"a flat index the patch below is <b>14 steps away</b>. It has no idea the "
             f"sequence is really a grid."),
            (GREEN, "The learned one worked out the grid.",
             f"Right <b>{lr:+.2f}</b> and below <b>{lb:+.2f}</b> are almost "
             f"<b>equal</b>, and everything far away sits at <b>{loth:+.2f}</b>. "
             f"Two-dimensional locality, learned from a one-dimensional index."),
        ])
        tip(f"ViT is handed nothing but a <b>flat index</b>, 0 to 196, and no hint that "
            f"the sequence is a grid. The fixed formula never discovers it. The "
            f"<b>learned</b> table does: right <b>{lr:+.2f}</b> against below "
            f"<b>{lb:+.2f}</b>. <b>The 2-D geometry was learned, not built in.</b>")

    with st.expander("Show the ViTEmbedding code"):
        st.code('''
class ViTEmbedding(nn.Module):
    def __init__(self, num_patches=196, embed_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbedding()                 # Section 3

        # One learnable vector, shared across every image
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # One learnable vector per slot: 196 patches + 1 CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x   = self.patch_embed(x)               # (B, 196, 768)
        cls = self.cls_token.expand(B, -1, -1)  # (B,   1, 768)
        x   = torch.cat([cls, x], dim=1)        # (B, 197, 768)
        return x + self.pos_embed               # added, not concatenated
''', language="python")


# ==============================================================================
# SECTION 5: SELF-ATTENTION
# ==============================================================================
elif num == 5:
    section_header(5, TOTAL, "Self-Attention over Patches",
                   "Every patch reads every other patch, in a single step")

    key_idea("A convolution sees a small neighbourhood. Self-attention lets <b>every "
             "patch look at every other patch at once</b>, so a patch of sky can relate "
             "to a wing tip 200 pixels away in <b>one</b> layer.")

    sub("Queries, Keys and Values")
    hint("Three vectors projected from every patch. A search engine is the right analogy.")
    grid([
        card("Query (Q)", "<b>What am I looking for?</b><br>Each patch broadcasts the kind "
             "of information it needs from the rest of the image.", PURPLE),
        card("Key (K)", "<b>What do I contain?</b><br>Each patch advertises its content so "
             "others can judge whether to attend to it.", BLUE),
        card("Value (V)", "<b>Here is my information.</b><br>The content actually passed "
             "along when a Query matches a Key.", GREEN),
    ], cols=3)

    html(f"""
    <div class="eq">
      <div class="eq-box" style="--c:{PURPLE};">
        <div class="eq-name">Q K<sup>T</sup></div>
        <div class="eq-desc">score every pair<br>of patches</div>
      </div>
      <div class="eq-op">/</div>
      <div class="eq-box" style="--c:{ORANGE};">
        <div class="eq-name">&#8730;d</div>
        <div class="eq-desc">&#8730;64 = 8<br>keeps scores small</div>
      </div>
      <div class="eq-op">&#8594;</div>
      <div class="eq-box" style="--c:{BLUE};">
        <div class="eq-name">softmax</div>
        <div class="eq-desc">turn scores into<br>weights summing to 1</div>
      </div>
      <div class="eq-op">x</div>
      <div class="eq-box" style="--c:{GREEN};">
        <div class="eq-name">V</div>
        <div class="eq-desc">weighted average<br>of the content</div>
      </div>
    </div>
    <p class="eq-note">Attention(Q, K, V) = softmax(Q K<sup>T</sup> / &#8730;d) V</p>""")

    keyrows([
        (BLUE, "Why 12 heads?",
         "768 dims split into <b>12 slices of 64</b>. One head can track colour while "
         "another tracks shape."),
        (BLUE, "The trade.",
         "Twelve cheap opinions beat one expensive one."),
        (ORANGE, "Why divide by root-d?",
         "Big dot products <b>saturate softmax</b>: one weight near 1, the rest near 0, "
         "gradients vanish. Dividing by 8 keeps it usable."),
        (GREEN, "Twelve layers, twelve readings.",
         "Each block re-reads the whole image through a different lens. Early layers tend "
         "to stay local, later layers range across the whole image."),
    ])

    sub("Try it: look inside a pretrained ViT's attention")
    hint("Real attention weights from <code>google/vit-base-patch16-224-in21k</code>. "
         "First run downloads about 330 MB.")

    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.image(img224, caption="Input image", use_container_width=True)
        layer_idx = st.slider("Encoder layer", 0, 11, 11)
        head_idx = st.slider("Attention head", 0, 11, 0)
    with c2:
        run = st.button("Extract attention", type="primary")
        hint("Layer 0 is the first block, layer 11 the last. Compare an early layer with "
             "a late one and watch how far attention reaches.")

    if run:
        with st.spinner("Running the pretrained model..."):
            vit = try_load_vit()
            if vit is None:
                st.stop()
            with torch.no_grad():
                out = vit(pixel_values=preprocess_image(img224))
            attns = out.attentions                      # 12 x (1, 12, 197, 197)

        dashboard([
            ("12", "Layers", "each re-reads the image", "#7DD3FC"),
            ("12", "Heads", "per layer, in parallel", "#C4B5FD"),
            ("197x197", "Attention map", "every token to every token", "#6EE7B7"),
            (f"L{layer_idx} H{head_idx}", "Showing", "move the sliders", GOLD),
        ])

        attn = attns[layer_idx][0, head_idx].numpy()    # (197, 197)
        cls_grid = attn[0, 1:].reshape(14, 14)
        arr = np.array(img224)

        hot = LinearSegmentedColormap.from_list("hot2", ["#FFFFFF", GOLD, ORANGE, RED])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        im0 = axes[0].imshow(attn[:20, :20], cmap="magma", vmin=0)
        axes[0].set_title("Attention matrix, first 20 tokens", loc="left")
        axes[0].set_xlabel("attends to")
        axes[0].set_ylabel("attends from")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(cls_grid, cmap=hot)
        axes[1].set_title("What CLS attends to", loc="left")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(arr)
        axes[2].imshow(upscale(normalize01(cls_grid)), cmap=hot, alpha=0.55)
        axes[2].set_title("Overlaid on the image", loc="left")
        axes[2].axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        sub("Attention rollout: all 12 layers at once")
        hint("One layer is one hop. <b>Rollout</b> multiplies all 12 layers together, "
             "adding the residual at each step, to show which patches shaped CLS overall.")

        rollout = torch.eye(197)
        for a in attns:
            m = a[0].mean(0)                     # average the 12 heads
            m = m + torch.eye(197)               # residual connection
            m = m / m.sum(dim=-1, keepdim=True)  # renormalise rows
            rollout = m @ rollout
        roll = normalize01(rollout[0, 1:].numpy().reshape(14, 14))

        fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4.2))
        ax2[0].imshow(roll, cmap=hot)
        ax2[0].set_title("Rollout map", loc="left")
        ax2[1].imshow(arr)
        ax2[1].imshow(upscale(roll), cmap=hot, alpha=0.55)
        ax2[1].set_title("Rollout over the image", loc="left")
        for a_ in ax2:
            a_.axis("off")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        tip("Rollout often lands on the main object, with <b>no segmentation labels "
            "ever provided</b>. It learned where to look from classification alone. "
            "Upload a photo with one clear subject.")

    with st.expander("Show the multi-head attention code"):
        st.code('''
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads          # 64
        self.scale     = self.head_dim ** -0.5           # 1/sqrt(64) = 1/8

        # One matrix produces Q, K and V together, then we split it
        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):                   # x: (B, 197, 768)
        B, N, D = x.shape
        H = self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)             # each (B, 12, 197, 64)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, 12, 197, 197)
        attn = attn.softmax(dim=-1)                     # rows sum to 1

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)               # (B, 197, 768), shape preserved
''', language="python")



# ==============================================================================
# SECTION 6: THE ENCODER BLOCK
# ==============================================================================
elif num == 6:
    section_header(6, TOTAL, "The Encoder Block",
                   "Attention plus an MLP, wrapped in residuals. Stacked twelve times.")

    key_idea("One block does two things: <b>mix information across patches</b> with "
             "attention, then <b>think about each patch on its own</b> with an MLP. Both "
             "sit inside a residual connection, so the <b>shape never changes</b>.")

    shapes([("(197, 768) in", BLUE), ("attention", PURPLE), ("MLP", ORANGE),
            ("(197, 768) out", GREEN)])

    sub("The block, in order")
    steps([
        "<b>LayerNorm</b>, then <b>multi-head attention</b>, then add the input back: "
        "<code>x = x + attn(norm(x))</code>. This is the only place patches talk "
        "to each other.",
        "<b>LayerNorm</b>, then the <b>MLP</b>, then add again: "
        "<code>x = x + mlp(norm(x))</code>. The MLP runs on each patch independently.",
        "The MLP widens 768 to <b>3072</b> (a ratio of 4), applies <b>GELU</b>, then "
        "projects back to 768. That expansion holds most of the block's parameters.",
    ])

    keyrows([
        (GREEN, "Why residuals?",
         "<code>x + f(x)</code> gives gradients a clear path back through 12 blocks. "
         "Without them, deep transformers simply do not train."),
        (BLUE, "Why pre-norm?",
         "ViT normalises <b>before</b> each sublayer. Post-norm normalises the sum, which "
         "runs large early in training. Modest at 12 layers, essential when deeper."),
        (ORANGE, "Attention mixes, the MLP does not.",
         "Attention is the only cross-patch operation. The MLP sees one patch at a time, "
         "so stacking blocks alternates mixing and refining."),
    ])

    sub("Try it: run one block")
    hint("A randomly initialised block on a random <code>(1, 197, 768)</code> tensor, the "
         "exact shape the real model carries. Watch the shape survive.")

    class MHA(nn.Module):
        def __init__(self, d=768, h=12):
            super().__init__()
            self.h, self.scale = h, (d // h) ** -0.5
            self.qkv = nn.Linear(d, d * 3, bias=False)
            self.proj = nn.Linear(d, d)

        def forward(self, x):
            B, N, D = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            a = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
            return self.proj((a @ v).transpose(1, 2).reshape(B, N, D))

    class Block(nn.Module):
        def __init__(self, d=768, ratio=4):
            super().__init__()
            self.norm1, self.attn = nn.LayerNorm(d), MHA(d)
            self.norm2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, d * ratio), nn.GELU(),
                                     nn.Linear(d * ratio, d))

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    if st.button("Run one encoder block", type="primary"):
        x = torch.randn(1, 197, 768)
        block = Block()
        with torch.no_grad():
            out = block(x)

        p_attn = sum(p.numel() for p in block.attn.parameters())
        p_mlp = sum(p.numel() for p in block.mlp.parameters())
        p_norm = sum(p.numel() for p in block.norm1.parameters()) * 2
        p_all = sum(p.numel() for p in block.parameters())

        dashboard([
            (str(tuple(x.shape)),   "Input",        "what the block receives", "#7DD3FC"),
            (str(tuple(out.shape)), "Output",       "identical, by design", "#6EE7B7"),
            (f"{p_all / 1e6:.1f}M", "Per block",    "parameters in one block", GOLD),
            (f"{p_all * 12 / 1e6:.0f}M", "All 12",  "the whole encoder", "#FDBA74"),
        ])
        table(["Sub-module", "Parameters", "Share"],
              [["Multi-head attention", f"{p_attn:,}", f"{p_attn / p_all * 100:.0f}%"],
               ["MLP (768 to 3072 to 768)", f"{p_mlp:,}", f"{p_mlp / p_all * 100:.0f}%"],
               ["LayerNorm x2", f"{p_norm:,}", f"{p_norm / p_all * 100:.1f}%"],
               ["<b>One block</b>", f"<b>{p_all:,}</b>", ""],
               ["<b>All 12 blocks</b>", f"<b>{p_all * 12:,}</b>", ""]],
              num_cols=(1, 2), highlight_rows=(4,))
        table(["Tensor", "Mean", "Std"],
              [["Input", f"{x.mean():+.4f}", f"{x.std():.4f}"],
               ["Output", f"{out.mean():+.4f}", f"{out.std():.4f}"]],
              num_cols=(1, 2))
        hint("The MLP holds about <b>two thirds</b> of every block's parameters, which "
             "surprises people who assume attention dominates. Attention is where the "
             "<i>compute</i> goes; the MLP is where the <i>weights</i> are.")

    with st.expander("Show the TransformerBlock code"):
        st.code('''
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),   # 768 -> 3072
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),   # 3072 -> 768
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm: normalise BEFORE the sublayer, add the residual after
        x = x + self.attn(self.norm1(x))    # patches exchange information
        x = x + self.mlp(self.norm2(x))     # each patch is refined alone
        return x                            # (B, 197, 768), unchanged shape
''', language="python")


# ==============================================================================
# SECTION 7: THE CLASSIFICATION HEAD
# ==============================================================================
elif num == 7:
    section_header(7, TOTAL, "The Classification Head",
                   "Twelve layers of work, read out through a single token")

    key_idea("After the last block, throw away all 196 patch vectors and keep "
             "<b>only the CLS vector</b>. One <code>Linear(768, classes)</code> turns it "
             "into scores. The head is under <b>1%</b> of the model.")

    shapes([("(197, 768)", BLUE), ("take CLS (768,)", RED),
            ("Linear", ORANGE), ("logits (classes,)", GREEN)])

    sub("From scores to probabilities")
    keyrows([
        (BLUE, "Logits.",
         "The raw outputs of the linear layer. Any real number, positive or negative, "
         "and they do not sum to anything meaningful."),
        (GREEN, "Softmax.",
         "Exponentiate each logit and divide by the total, giving positive numbers that "
         "<b>sum to 1</b>. The ordering never changes, so the prediction is the same; "
         "only the interpretation becomes a probability."),
        (ORANGE, "Pretrain vs fine-tune.",
         "The paper pretrains with an <b>MLP head</b>, then swaps in a <b>single "
         "Linear</b> for fine-tuning. Simpler is enough once the encoder is good."),
        (PURPLE, "Why only CLS?",
         "The one token carrying no image content, so it is free to be a summary. "
         "Averaging all 196 patches also works, and some later models prefer it."),
    ])

    sub("Try it: run the head")
    hint("A random encoder output pushed through the head, to compare logits with "
         "probabilities. Ten classes keeps the chart readable.")

    c1, c2 = st.columns([1, 1.6])
    with c1:
        finetune = st.toggle("Fine-tuning head (single Linear)", value=True)
        n_classes = st.select_slider("Classes", options=[10, 100, 1000], value=10)
    with c2:
        run = st.button("Run the head", type="primary")
        hint("Turn the toggle off to use the larger pretraining MLP head and watch the "
             "parameter count jump.")

    if run:
        d, hidden = 768, 3072
        head = (nn.Linear(d, n_classes) if finetune else
                nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_classes)))
        cls_vec = torch.randn(1, d)
        with torch.no_grad():
            logits = head(cls_vec)[0]
            probs = logits.softmax(-1)

        k = min(8, n_classes)
        top = logits.topk(k)
        names = [f"class {i}" for i in top.indices.tolist()]
        lg = top.values.numpy()
        pr = probs[top.indices].numpy()

        p_head = sum(p.numel() for p in head.parameters())
        dashboard([
            (f"{p_head:,}", "Head params",
             "linear" if finetune else "MLP", GOLD),
            (f"{p_head / VIT_BASE_TOTAL * 100:.2f}%", "Of ViT-Base",
             "the head is tiny", "#7DD3FC"),
            (f"{n_classes}", "Classes", "output scores", "#C4B5FD"),
            (f"{probs.sum():.2f}", "Probabilities", "always sum to 1", "#6EE7B7"),
        ])

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
        axes[0].bar(names, lg, color=BLUE)
        axes[0].axhline(0, color=MUTED, lw=.8, ls="--")
        axes[0].set_title("Raw logits, before softmax", loc="left")
        axes[0].set_ylabel("logit")
        bars = axes[1].bar(names, pr, color=GREEN)
        for b, v in zip(bars, pr):
            axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + .004,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        axes[1].set_title("After softmax, sums to 1", loc="left")
        axes[1].set_ylabel("probability")
        for ax in axes:
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            style_axes(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        table(["Component", "Parameters"],
              [[f"Head ({'linear, fine-tuning' if finetune else 'MLP, pretraining'})",
                f"{p_head:,}"],
               ["Encoder, embeddings and the rest",
                f"{VIT_BASE_TOTAL - p_head:,}"],
               ["<b>ViT-Base total</b>", f"<b>{VIT_BASE_TOTAL:,}</b>"]],
              num_cols=(1,), highlight_rows=(2,))
        hint("The encoder does nearly all the work. That is why <b>fine-tuning</b> is "
             "normal: keep the 86M pretrained weights, swap this tiny layer, train.")

    with st.expander("Show the head code"):
        st.code('''
class MLPHead(nn.Module):
    """Pretraining uses the MLP; fine-tuning swaps in a single Linear."""
    def __init__(self, embed_dim=768, num_classes=1000, hidden_dim=3072,
                 finetune=False):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        if finetune:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x):            # x: (B, 197, 768), the encoder output
        cls = self.norm(x[:, 0])     # keep token 0 only -> (B, 768)
        return self.head(cls)        # (B, num_classes) logits, NOT probabilities

# Softmax belongs in the loss, not the model:
# nn.CrossEntropyLoss expects raw logits and applies log-softmax itself.
''', language="python")


# ==============================================================================
# SECTION 8: THE FULL MODEL
# ==============================================================================
elif num == 8:
    section_header(8, TOTAL, "The Full Model",
                   "Every piece from sections 2 to 7, assembled and run end to end")

    key_idea("ViT is four stages: <b>embed the patches</b>, <b>add CLS and position</b>, "
             "<b>twelve encoder blocks</b>, <b>read CLS through a linear head</b>. "
             "That is the whole architecture.")

    sub("The whole forward pass")
    html(f"""
    <div class="pipe">
      <div class="pipe-row">
        <div class="pipe-num" style="--c:{BLUE};">1</div>
        <div class="pipe-body" style="--c:{BLUE};">
          <div class="pipe-what">Patch embedding</div>
          <div class="pipe-why"><code>Conv2d(3, 768, 16, stride=16)</code> tiles and
          projects in one step.
          <span class="chain">(1, 3, 224, 224) &#8594; (1, 196, 768)</span></div>
        </div>
      </div>
      <div class="pipe-link"><span>&#8595;</span></div>
      <div class="pipe-row">
        <div class="pipe-num" style="--c:{RED};">2</div>
        <div class="pipe-body" style="--c:{RED};">
          <div class="pipe-what">CLS and position</div>
          <div class="pipe-why">Prepend the summary token, add the learned positions.
          <span class="chain">(1, 196, 768) &#8594; (1, 197, 768)</span></div>
        </div>
      </div>
      <div class="pipe-link"><span>&#8595;</span></div>
      <div class="pipe-row">
        <div class="pipe-num" style="--c:{ORANGE};">3</div>
        <div class="pipe-body" style="--c:{ORANGE};">
          <div class="pipe-what">Twelve encoder blocks</div>
          <div class="pipe-why">Attention then MLP, residuals throughout. Shape is
          identical at every layer.
          <span class="chain">(1, 197, 768) &#8594; (1, 197, 768)</span></div>
        </div>
      </div>
      <div class="pipe-link"><span>&#8595;</span></div>
      <div class="pipe-row">
        <div class="pipe-num" style="--c:{GREEN};">4</div>
        <div class="pipe-body" style="--c:{GREEN};">
          <div class="pipe-what">LayerNorm and head</div>
          <div class="pipe-why">Keep token 0, project to class scores.
          <span class="chain">(1, 197, 768) &#8594; (1, 768) &#8594; (1, classes)</span></div>
        </div>
      </div>
    </div>""")

    sub("Try it: build ViT and push your image through")
    hint("A full randomly initialised ViT-Base on your image. Predictions are "
         "meaningless; the <b>shapes and the parameter count</b> are the point.")

    class ViT(nn.Module):
        def __init__(self, img=224, patch=16, n_classes=1000, d=768, depth=12, heads=12):
            super().__init__()
            n_patches = (img // patch) ** 2
            self.patch_embed = nn.Conv2d(3, d, patch, patch)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d))
            self.blocks = nn.Sequential(*[Block8(d, heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, n_classes)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            x = x + self.pos_embed
            x = self.norm(self.blocks(x))
            return self.head(x[:, 0])

    class MHA8(nn.Module):
        def __init__(self, d=768, h=12):
            super().__init__()
            self.h, self.scale = h, (d // h) ** -0.5
            self.qkv = nn.Linear(d, d * 3, bias=False)
            self.proj = nn.Linear(d, d)

        def forward(self, x):
            B, N, D = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            a = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
            return self.proj((a @ v).transpose(1, 2).reshape(B, N, D))

    class Block8(nn.Module):
        def __init__(self, d=768, h=12, ratio=4):
            super().__init__()
            self.norm1, self.attn = nn.LayerNorm(d), MHA8(d, h)
            self.norm2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, d * ratio), nn.GELU(),
                                     nn.Linear(d * ratio, d))

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.image(img224, caption="Input image", use_container_width=True)
    with c2:
        run = st.button("Run the full ViT forward pass", type="primary")

    if run:
        with st.spinner("Assembling ViT-Base and running the forward pass..."):
            model = ViT(n_classes=1000)
            with torch.no_grad():
                logits = model(preprocess_image(img224))

        p_emb = (model.patch_embed.weight.numel() + model.patch_embed.bias.numel()
                 + model.cls_token.numel() + model.pos_embed.numel())
        p_blocks = sum(p.numel() for p in model.blocks.parameters())
        p_head = sum(p.numel() for p in model.head.parameters())
        p_total = sum(p.numel() for p in model.parameters())

        dashboard([
            (str(tuple(logits.shape)),      "Output",   "one score per class", "#7DD3FC"),
            (f"{p_total / 1e6:.1f}M",       "Total",    "parameters in ViT-Base", GOLD),
            (f"{p_blocks / p_total * 100:.0f}%", "Encoder", "where the weights live",
             "#FDBA74"),
            (f"{p_head / 1e3:.0f}K",        "Head",     "under 1 percent", "#6EE7B7"),
        ])

        table(["Stage", "Parameters", "Share"],
              [["Patch embedding, CLS, position", f"{p_emb:,}",
                f"{p_emb / p_total * 100:.1f}%"],
               ["12 encoder blocks", f"{p_blocks:,}", f"{p_blocks / p_total * 100:.1f}%"],
               ["Final LayerNorm and head", f"{p_head + 2 * 768:,}",
                f"{(p_head + 2 * 768) / p_total * 100:.1f}%"],
               ["<b>Total</b>", f"<b>{p_total:,}</b>", "100%"]],
              num_cols=(1, 2), highlight_rows=(1,))
        keyrows([
            (NAVY, "ViT-Base scale.",
             "About <b>86.5M</b> parameters here. Published figures sit near 86.4M; the "
             "gap is bias terms and head size, not a different architecture."),
            (ORANGE, "Where the weights live.",
             f"<b>{p_blocks / p_total * 100:.1f}%</b> in the twelve encoder blocks. "
             "The head is under <b>1%</b>."),
            (RED, "The predictions are noise.",
             "This model has never seen a label."),
        ])

    with st.expander("Show the complete ViT class"):
        st.code('''
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2            # 196

        # 1. patch embedding + 2. CLS and position
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout     = nn.Dropout(dropout)

        # 3. the encoder
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # 4. readout
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):                                 # (B, 3, 224, 224)
        B = x.shape[0]
        x   = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, 196, 768)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                       # (B, 197, 768)
        x   = self.dropout(x + self.pos_embed)
        x   = self.norm(self.blocks(x))                        # (B, 197, 768)
        return self.head(x[:, 0])                              # (B, num_classes)

model = ViT()
total = sum(p.numel() for p in model.parameters())    # 86,415,592
''', language="python")


# ==============================================================================
# SECTION 9: TRAINING IN PRACTICE
# ==============================================================================
elif num == 9:
    section_header(9, TOTAL, "Training in Practice",
                   "Why you will almost certainly fine-tune rather than train from scratch")

    key_idea("ViT has <b>no built-in sense of locality</b>, so it must learn spatial "
             "structure from examples. That makes it <b>data hungry</b>. In practice you "
             "start from a pretrained checkpoint and fine-tune.")

    sub("Two ways to train, and only one you will likely use")
    table(
        ["", f"{tag('Fine-tuning', GREEN)}", f"{tag('From scratch', RED)}"],
        [
            ["Data needed", "Thousands of labelled images.",
             "<b>14M or more</b> (ImageNet-21k, JFT-300M)."],
            ["Hardware", "One GPU, minutes to hours.", "Many GPUs, days to weeks."],
            ["Learning rate", "Small: <code>1e-5</code> to <code>1e-4</code>.",
             "Larger: around <code>1e-3</code>."],
            ["Beats a CNN?", "Usually yes.", "Only once the data is genuinely large."],
            ["Who does this", "Essentially every applied project.",
             "Labs with a cluster and a reason."],
        ],
    )

    keyrows([
        (ORANGE, "Augmentation is not optional.",
         "With no locality bias, ViT overfits small datasets fast. RandAugment, Mixup, "
         "CutMix and random erasing do much of the work the architecture does not."),
        (BLUE, "Warmup then cosine decay.",
         "Transformers are unstable in the first few hundred steps. Warm the learning "
         "rate up over a few epochs, then decay it smoothly to near zero."),
        (PURPLE, "Clip the gradients.",
         "<code>clip_grad_norm_(..., max_norm=1.0)</code> is standard for ViT training "
         "and prevents a single bad batch from wrecking the run."),
        (GREEN, "Changing resolution?",
         "Position embeddings are tied to the 14x14 grid and must be <b>interpolated</b>. "
         "A standard utility, not a research problem."),
    ])

    sub("Try it: one training step, end to end")
    hint("A real forward pass, loss, and backward pass on a small random batch, so you "
         "can see gradients actually appear. CPU-sized on purpose.")

    class MHA9(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.h, self.scale = h, (d // h) ** -0.5
            self.qkv = nn.Linear(d, d * 3, bias=False)
            self.proj = nn.Linear(d, d)

        def forward(self, x):
            B, N, D = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            a = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
            return self.proj((a @ v).transpose(1, 2).reshape(B, N, D))

    class Block9(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.norm1, self.attn = nn.LayerNorm(d), MHA9(d, h)
            self.norm2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class ViT9(nn.Module):
        """A deliberately small ViT so one training step runs on a CPU."""
        def __init__(self, n_classes=10, d=192, depth=4, heads=3, img=224, patch=16):
            super().__init__()
            n_patches = (img // patch) ** 2
            self.patch_embed = nn.Conv2d(3, d, patch, patch)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d))
            self.blocks = nn.Sequential(*[Block9(d, heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, n_classes)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
            return self.head(self.norm(self.blocks(x))[:, 0])

    if st.button("Run one training step", type="primary"):
        with st.spinner("Forward, loss, backward..."):
            small = ViT9(n_classes=10, d=192, depth=4, heads=3)
            x = torch.randn(4, 3, 224, 224)
            y = torch.randint(0, 10, (4,))
            crit = nn.CrossEntropyLoss(label_smoothing=0.1)
            opt = torch.optim.AdamW(small.parameters(), lr=1e-3, weight_decay=0.05)

            logits = small(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(small.parameters(), max_norm=1.0)
            opt.step()

        dashboard([
            (str(tuple(logits.shape)), "Logits",    "batch of 4, ten classes", "#7DD3FC"),
            (f"{loss.item():.2f}",     "Loss",      "chance is 2.30", GOLD),
            (f"{gnorm:.1f}",           "Grad norm", "clipped to 1.0", "#FDBA74"),
            (f"{sum(p.numel() for p in small.parameters()) / 1e6:.1f}M", "Parameters",
             "this small ViT", "#6EE7B7"),
        ])

        keyrows([
            (BLUE, "Chance level.",
             f"Ten classes, untrained, so <code>ln(10) = {math.log(10):.2f}</code>. "
             f"Measured <b>{loss.item():.2f}</b>, which scatters either side with only "
             f"4 images."),
            (GREEN, "What that proves.",
             "The wiring is right. <b>Nothing has been learned.</b>"),
            (ORANGE, "Why clip the gradient?",
             f"The raw gradient norm here was <b>{gnorm:.3f}</b>. Clipping at 1.0 caps "
             "how far any single step can move the weights."),
        ])
        tip("A deliberately small ViT (192-dim, 4 layers) so it runs on a CPU in "
            "seconds. <b>Real ViT-Base training is the same five lines</b>, with 86M "
            "parameters and a very large dataset behind them.")

    with st.expander("Show the fine-tuning recipe"):
        st.code('''
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ViTForImageClassification

# Start from pretrained weights and replace the head for your classes.
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10,
    ignore_mismatched_sizes=True,      # the old head has the wrong shape, drop it
)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

for images, labels in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(images).logits, labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # standard for ViT
    optimizer.step()
scheduler.step()

# Notes that matter more than the code:
#   - augment aggressively (RandAugment, Mixup, CutMix); ViT overfits small data
#   - warm the LR up for the first few epochs, then let cosine decay take over
#   - changing input resolution means interpolating the position embeddings
''', language="python")



# -- Previous / Next -----------------------------------------------------------
st.divider()
p, _, n = st.columns([1, 2, 1])
with p:
    st.button("<-  Previous", on_click=go, args=(-1,), disabled=num == 1,
              use_container_width=True, key="prev_btn")
with n:
    st.button("Next  ->", on_click=go, args=(1,), disabled=num == TOTAL,
              type="primary", use_container_width=True, key="next_btn")
