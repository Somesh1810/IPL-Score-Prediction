"""
app.py  –  IPL Score Prediction  |  Redesigned UI
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

:root {
    --green:  #00ff87;
    --green2: #00cc6a;
    --red:    #ff3b5c;
    --bg:     #080b10;
    --bg2:    #0d1117;
    --bg3:    #131920;
    --border: #1e2d1e;
    --text:   #c8d8c8;
    --muted:  #4a6a4a;
}
* { box-sizing: border-box; }
html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Barlow', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1120px !important; }

.topbar {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--green);
    padding: 1.2rem 0 0.8rem 0; margin-bottom: 2rem;
}
.logo-text {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800; font-size: 2.2rem;
    color: var(--green); letter-spacing: 0.02em; text-transform: uppercase;
}
.logo-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase;
}
.live-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem; background: var(--red); color: white;
    padding: 0.2rem 0.6rem; border-radius: 2px; letter-spacing: 0.1em;
    animation: blink 1.5s step-end infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.sec-header {
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    color: var(--green); letter-spacing: 0.2em; text-transform: uppercase;
    border-left: 2px solid var(--green); padding-left: 0.6rem;
    margin-bottom: 0.8rem; margin-top: 1.4rem;
}
.panel {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 4px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.panel::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: linear-gradient(90deg, var(--green), transparent);
}

div[data-baseweb="select"] > div {
    background: var(--bg3) !important; border-color: var(--border) !important;
    border-radius: 3px !important; font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important; color: var(--text) !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--green) !important; box-shadow: 0 0 0 1px var(--green) !important;
}
div[data-testid="stNumberInput"] input {
    background: var(--bg3) !important; border-color: var(--border) !important;
    border-radius: 3px !important; color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important; font-size: 1.1rem !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--green) !important; box-shadow: 0 0 0 1px var(--green) !important;
}
div[data-testid="stSlider"] > div > div > div > div { background: var(--green) !important; }
div[data-testid="stSlider"] > div > div > div { background: var(--border) !important; }

label, .stSlider label, .stSelectbox label, div[data-testid="stNumberInput"] label {
    font-family: 'Barlow Condensed', sans-serif !important; font-size: 0.78rem !important;
    font-weight: 600 !important; color: var(--muted) !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
}

div[data-testid="stButton"] > button {
    background: var(--green) !important; color: var(--bg) !important;
    font-family: 'Barlow Condensed', sans-serif !important; font-weight: 800 !important;
    font-size: 1.3rem !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; border: none !important;
    border-radius: 3px !important; padding: 0.7rem 2.5rem !important; width: 100% !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stButton"] > button:hover {
    background: var(--green2) !important; transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(0,255,135,0.25) !important;
}

.scoreboard {
    background: var(--bg); border: 1px solid var(--green); border-radius: 4px;
    padding: 2rem 2rem 1.5rem; text-align: center;
    box-shadow: 0 0 40px rgba(0,255,135,0.08), inset 0 0 60px rgba(0,255,135,0.02);
    margin-top: 0.5rem;
}
.scoreboard-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    color: var(--muted); letter-spacing: 0.25em; text-transform: uppercase; margin-bottom: 0.5rem;
}
.scoreboard-score {
    font-family: 'Share Tech Mono', monospace; font-size: 5.5rem;
    color: var(--green); line-height: 1;
    text-shadow: 0 0 30px rgba(0,255,135,0.4);
}
.scoreboard-range {
    font-family: 'Barlow Condensed', sans-serif; font-size: 1.1rem;
    color: var(--muted); letter-spacing: 0.1em; margin-top: 0.3rem;
}
.scoreboard-range span { color: var(--text); }
.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem; margin-top: 1.2rem; }
.stat-tile {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 3px; padding: 0.8rem 0.6rem; text-align: center;
}
.stat-val { font-family: 'Share Tech Mono', monospace; font-size: 1.5rem; color: var(--green); display: block; }
.stat-lbl {
    font-family: 'Barlow Condensed', sans-serif; font-size: 0.65rem;
    color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase;
    margin-top: 0.2rem; display: block;
}
.progress-strip-wrap { margin-top: 1.4rem; }
.progress-strip-label {
    display: flex; justify-content: space-between;
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    color: var(--muted); margin-bottom: 0.3rem;
}
.progress-strip-bg {
    height: 6px; background: var(--bg3); border-radius: 1px;
    overflow: hidden; border: 1px solid var(--border);
}
.progress-strip-fill {
    height: 100%; background: linear-gradient(90deg, var(--green2), var(--green));
    border-radius: 1px; box-shadow: 0 0 8px rgba(0,255,135,0.5);
}
.matchup {
    display: flex; align-items: center; justify-content: space-between;
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 3px; padding: 0.7rem 1.2rem; margin-bottom: 1rem;
}
.matchup-team {
    font-family: 'Barlow Condensed', sans-serif; font-weight: 700;
    font-size: 1.05rem; color: var(--text); letter-spacing: 0.04em; text-transform: uppercase;
}
.matchup-vs { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: var(--muted); letter-spacing: 0.15em; }
.bat-tag {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    background: rgba(0,255,135,0.12); color: var(--green);
    border: 1px solid rgba(0,255,135,0.25); padding: 0.1rem 0.4rem;
    border-radius: 2px; letter-spacing: 0.1em; margin-left: 0.5rem;
}
.bowl-tag {
    font-family: 'Share Tech Mono', monospace; font-size: 0.55rem;
    background: rgba(255,59,92,0.12); color: var(--red);
    border: 1px solid rgba(255,59,92,0.25); padding: 0.1rem 0.4rem;
    border-radius: 2px; letter-spacing: 0.1em; margin-right: 0.5rem;
}
section[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    if not os.path.exists("model.pkl"):
        st.error("model.pkl not found. Run: python generate_data.py && python train_model.py")
        st.stop()
    model   = joblib.load("model.pkl")
    le_bat  = joblib.load("le_bat.pkl")
    le_bowl = joblib.load("le_bowl.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, le_bat, le_bowl, meta


model, le_bat, le_bowl, meta = load_artifacts()
TEAMS = meta["teams"]

st.markdown("""
<div class="topbar">
  <div>
    <div class="logo-text">🏏 IPL Score Predictor</div>
    <div class="logo-sub">ML-Powered innings forecast</div>
  </div>
  <div class="live-badge">● LIVE MODEL</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="sec-header">01 — Match Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        batting_team = st.selectbox("🏏 Batting Team", TEAMS, key="bat")
    with c2:
        bowl_opts = [t for t in TEAMS if t != batting_team]
        bowling_team = st.selectbox("🎯 Bowling Team", bowl_opts, key="bowl")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-header">02 — Live Situation</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    with ca:
        overs = st.slider("Overs", min_value=5, max_value=19, value=10)
    with cb:
        current_score = st.number_input("Runs", min_value=10, max_value=260, value=87)
    with cc:
        wickets = st.slider("Wickets", min_value=0, max_value=9, value=3)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-header">03 — Last 5 Overs</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    cd, ce = st.columns(2)
    with cd:
        runs_last5 = st.number_input("Runs (last 5)", min_value=0, max_value=120, value=45)
    with ce:
        wickets_last5 = st.number_input("Wickets (last 5)", min_value=0, max_value=5, value=1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("⚡ Predict Final Score")

with right:
    st.markdown('<div class="sec-header">04 — Prediction Output</div>', unsafe_allow_html=True)

    if predict:
        if wickets_last5 > wickets:
            st.warning("⚠️ Last-5 wickets cannot exceed total wickets.")
        elif current_score < runs_last5:
            st.warning("⚠️ Current score cannot be less than last-5 runs.")
        else:
            run_rate        = current_score / max(overs, 1)
            wickets_left    = 10 - wickets
            balls_remaining = (20 - overs) * 6
            rr_last5        = runs_last5 / 5.0
            pressure_index  = wickets / max(overs, 1)
            remaining_overs = 20 - overs

            bat_enc  = le_bat.transform([batting_team])[0]
            bowl_enc = le_bowl.transform([bowling_team])[0]

            input_df = pd.DataFrame([{
                "batting_team": bat_enc, "bowling_team": bowl_enc,
                "overs": overs, "current_score": current_score, "wickets": wickets,
                "runs_last5": runs_last5, "wickets_last5": wickets_last5,
                "run_rate": run_rate, "wickets_left": wickets_left,
                "balls_remaining": balls_remaining, "rr_last5": rr_last5,
                "pressure_index": pressure_index,
            }])

            prediction     = int(model.predict(input_df)[0])
            low            = max(current_score + 5, prediction - 10)
            high           = prediction + 10
            remaining_runs = max(prediction - current_score, 0)
            req_rr         = round(remaining_runs / max(remaining_overs, 1), 2)
            progress_pct   = min(int((current_score / max(prediction, 1)) * 100), 100)

            st.markdown(f"""
            <div class="matchup">
              <div><span class="matchup-team">{batting_team}</span><span class="bat-tag">BAT</span></div>
              <div class="matchup-vs">VS</div>
              <div><span class="bowl-tag">BOWL</span><span class="matchup-team">{bowling_team}</span></div>
            </div>
            <div class="scoreboard">
              <div class="scoreboard-label">Predicted Final Score</div>
              <div class="scoreboard-score">{prediction}</div>
              <div class="scoreboard-range">Range &nbsp;<span>{low} – {high}</span></div>
              <div class="stat-row">
                <div class="stat-tile"><span class="stat-val">{run_rate:.1f}</span><span class="stat-lbl">Curr RR</span></div>
                <div class="stat-tile"><span class="stat-val">{req_rr}</span><span class="stat-lbl">Req RR</span></div>
                <div class="stat-tile"><span class="stat-val">{balls_remaining}</span><span class="stat-lbl">Balls Left</span></div>
                <div class="stat-tile"><span class="stat-val">{wickets_left}</span><span class="stat-lbl">Wkts Left</span></div>
              </div>
              <div class="progress-strip-wrap">
                <div class="progress-strip-label">
                  <span>INNINGS PROGRESS</span>
                  <span>{current_score} / {prediction} &nbsp;({progress_pct}%)</span>
                </div>
                <div class="progress-strip-bg">
                  <div class="progress-strip-fill" style="width:{progress_pct}%"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="scoreboard" style="opacity:0.4">
          <div class="scoreboard-label">Awaiting Input</div>
          <div class="scoreboard-score" style="font-size:3rem;letter-spacing:0.2em">— — —</div>
          <div class="scoreboard-range">Fill in match details and click Predict</div>
          <div class="stat-row">
            <div class="stat-tile"><span class="stat-val">—</span><span class="stat-lbl">Curr RR</span></div>
            <div class="stat-tile"><span class="stat-val">—</span><span class="stat-lbl">Req RR</span></div>
            <div class="stat-tile"><span class="stat-val">—</span><span class="stat-lbl">Balls Left</span></div>
            <div class="stat-tile"><span class="stat-val">—</span><span class="stat-lbl">Wkts Left</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🤖 Model Info")
    st.markdown(f"**Algorithm:** `{meta['best_model']}`")
    st.markdown(f"**MAE:** {meta['metrics']['MAE']} runs")
    st.markdown(f"**RMSE:** {meta['metrics']['RMSE']} runs")
    st.markdown(f"**CV MAE:** {meta['metrics']['CV_MAE']} runs")
    st.divider()
    st.markdown("### Features")
    for f in meta["features"]:
        st.markdown(f"- `{f}`")
    st.divider()
    st.caption("Built by Somesh M · Statistics & Data Science")