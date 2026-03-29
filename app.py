"""
IPL Win Probability Predictor — Streamlit App
"""

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}
.block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1400px;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.app-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #f0c040 0%, #ff6b35 60%, #e8355a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 0;
}
.app-subtitle {
    font-size: 0.9rem;
    color: #6b7280;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Cards ── */
.card {
    background: #13182b;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 1.2rem;
}

/* ── Probability Display ── */
.prob-display {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(145deg, #13182b, #1a2035);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
}
.prob-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 7rem;
    letter-spacing: 2px;
    line-height: 1;
}
.prob-team {
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0.4rem 0 0.2rem;
    letter-spacing: 0.5px;
}
.prob-label {
    font-size: 0.78rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── Progress bar custom ── */
.split-bar-wrap {
    background: #1e2640;
    border-radius: 50px;
    height: 18px;
    overflow: hidden;
    margin: 0.8rem 0;
    display: flex;
}
.split-bar-left {
    height: 100%;
    background: linear-gradient(90deg, #f0c040, #ff6b35);
    border-radius: 50px 0 0 50px;
    transition: width 0.6s ease;
}
.split-bar-right {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    border-radius: 0 50px 50px 0;
    transition: width 0.6s ease;
    flex: 1;
}

/* ── Insight box ── */
.insight-box {
    background: rgba(240, 192, 64, 0.06);
    border-left: 3px solid #f0c040;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #c9d0e0;
    margin: 0.8rem 0;
}
.insight-box.danger {
    background: rgba(239, 68, 68, 0.06);
    border-left-color: #ef4444;
}
.insight-box.safe {
    background: rgba(34, 197, 94, 0.06);
    border-left-color: #22c55e;
}

/* ── Stat chips ── */
.stat-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin: 0.8rem 0;
}
.stat-chip {
    background: #1e2640;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.45rem 0.9rem;
    font-size: 0.78rem;
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.stat-chip-label { color: #6b7280; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1px; }
.stat-chip-value { color: #e8eaf0; font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.9rem; }

/* ── Momentum badge ── */
.momentum-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.35rem 0.85rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.momentum-high { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.momentum-mid  { background: rgba(234,179,8,0.15);  color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
.momentum-low  { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* ── Section divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 1.5rem 0;
}

/* ── Streamlit widget overrides ── */
.stSlider > div > div { background: #1e2640 !important; }
div[data-testid="stSelectbox"] > div { background: #13182b !important; border-color: rgba(255,255,255,0.1) !important; }
div[data-testid="stNumberInput"] input { background: #13182b !important; border-color: rgba(255,255,255,0.1) !important; color: #e8eaf0 !important; }
.stButton > button {
    background: linear-gradient(135deg, #f0c040, #ff6b35) !important;
    color: #0a0e1a !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 2rem !important;
    border-radius: 10px !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #9ca3af !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_importance():
    try:
        return pd.read_csv("data/feature_importance.csv")
    except FileNotFoundError:
        return None

model = load_model()
importance_df = load_importance()

IPL_TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
    "Rajasthan Royals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans",
]

NUMERIC_FEATURES = [
    "runs_left", "balls_left", "wickets_left",
    "current_run_rate", "required_run_rate",
    "pressure_index", "innings_progress",
]
CATEGORICAL_FEATURES = ["batting_team", "bowling_team"]


# ── Helper: predict ────────────────────────────────────────────────────────────
def predict_win_probability(
    batting_team, bowling_team, target, score, balls_played, wickets_fallen
):
    balls_left = 120 - balls_played
    runs_left = max(target - score, 0)
    wickets_left = max(10 - wickets_fallen, 0)
    overs_done = balls_played / 6

    # Edge cases
    if runs_left <= 0:
        return 1.0
    if wickets_left <= 0 or balls_left <= 0:
        return 0.0

    crr = score / overs_done if overs_done > 0 else 0.0
    rrr = min((runs_left / (balls_left / 6)), 36.0) if balls_left > 0 else 36.0
    pressure_index = np.clip(rrr - crr, -10, 20)
    innings_progress = (120 - balls_left) / 120

    row = pd.DataFrame([{
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "current_run_rate": round(crr, 3),
        "required_run_rate": round(rrr, 3),
        "pressure_index": pressure_index,
        "innings_progress": innings_progress,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
    }])

    prob = model.predict_proba(row[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0][1]
    return float(prob)


def get_confidence(prob):
    dist = abs(prob - 0.5)
    if dist > 0.35:
        return "Very High", "momentum-high"
    elif dist > 0.2:
        return "High", "momentum-high"
    elif dist > 0.1:
        return "Moderate", "momentum-mid"
    else:
        return "Low", "momentum-low"


def get_insight(batting_team, prob, rrr, crr, wickets_left, balls_left):
    lines = []
    if prob >= 0.75:
        lines.append(f"🟢 <b>{batting_team}</b> are firmly in control of this chase.")
    elif prob >= 0.55:
        lines.append(f"🟡 <b>{batting_team}</b> hold a slight edge but the game is alive.")
    elif prob >= 0.40:
        lines.append(f"⚔️ This is a genuine contest — anyone's match to win.")
    else:
        lines.append(f"🔴 The bowling side is dominating. <b>{batting_team}</b> need something special.")

    if rrr > crr + 3:
        lines.append(f"Required run rate ({rrr:.1f}) is significantly above current rate ({crr:.1f}) — pressure is mounting.")
    elif rrr < crr - 2:
        lines.append(f"With RRR ({rrr:.1f}) well below current rate ({crr:.1f}), the chasing team can afford to be patient.")

    if wickets_left <= 3:
        lines.append("⚠️ Only 3 wickets remaining — a collapse could end the chase quickly.")
    elif wickets_left >= 8:
        lines.append("Plenty of batting depth remaining — resources are in good shape.")

    overs_left = balls_left / 6
    if overs_left <= 5 and balls_left > 0:
        lines.append(f"Death overs ahead ({overs_left:.1f} left). Boundaries and smart placement will be decisive.")

    return " ".join(lines)


def get_momentum(prob, rrr, crr, wickets_left):
    score = 0
    if prob > 0.6: score += 2
    elif prob > 0.4: score += 1
    if rrr < crr: score += 2
    elif rrr < crr + 3: score += 1
    if wickets_left >= 7: score += 2
    elif wickets_left >= 4: score += 1
    if score >= 5:
        return "Strong 📈", "momentum-high"
    elif score >= 3:
        return "Neutral ↔", "momentum-mid"
    else:
        return "Under Pressure 📉", "momentum-low"


# ── Win probability curve data ─────────────────────────────────────────────────
def compute_probability_curve(batting_team, bowling_team, target, balls_played, wickets_fallen):
    scores = list(range(0, target + 5, max(1, target // 40)))
    probs = []
    for s in scores:
        p = predict_win_probability(batting_team, bowling_team, target, s, balls_played, wickets_fallen)
        probs.append(p * 100)
    return scores, probs


# ── App Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1 class="app-title">🏏 IPL WIN PREDICTOR</h1>
    <span class="app-subtitle">T20 Live Match Analytics</span>
</div>
""", unsafe_allow_html=True)

# ── Layout: 2 columns ──────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")

# ══ LEFT: Inputs ═══════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="card-title">Match Situation</div>', unsafe_allow_html=True)

    batting_team = st.selectbox("Batting Team (Chasing)", IPL_TEAMS, index=0)
    available_bowling = [t for t in IPL_TEAMS if t != batting_team]
    bowling_team = st.selectbox("Bowling Team (Defending)", available_bowling, index=0)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    target = st.number_input("Target Score", min_value=50, max_value=300, value=175, step=1)
    score  = st.number_input("Current Score", min_value=0, max_value=target - 1, value=80, step=1)

    col1, col2 = st.columns(2)
    with col1:
        overs = st.slider("Overs Completed", 0.0, 19.5, 10.0, step=0.5,
                          help="Each 0.5 = 3 balls")
    with col2:
        wickets = st.slider("Wickets Fallen", 0, 10, 3)

    balls_played = int(overs * 6)

    # Live derived stats
    balls_left = 120 - balls_played
    runs_left  = max(target - score, 0)
    crr = round(score / (balls_played / 6), 2) if balls_played > 0 else 0.0
    rrr = round(min(runs_left / (balls_left / 6), 36.0), 2) if balls_left > 0 else 99.0

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-chip">
            <span class="stat-chip-label">Runs Left</span>
            <span class="stat-chip-value">{runs_left}</span>
        </div>
        <div class="stat-chip">
            <span class="stat-chip-label">Balls Left</span>
            <span class="stat-chip-value">{balls_left}</span>
        </div>
        <div class="stat-chip">
            <span class="stat-chip-label">Wickets Left</span>
            <span class="stat-chip-value">{10-wickets}</span>
        </div>
        <div class="stat-chip">
            <span class="stat-chip-label">CRR</span>
            <span class="stat-chip-value">{crr}</span>
        </div>
        <div class="stat-chip">
            <span class="stat-chip-label">RRR</span>
            <span class="stat-chip-value">{rrr}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("⚡ PREDICT WIN PROBABILITY")

    # What-if section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">What-If Simulator</div>', unsafe_allow_html=True)
    whatif_runs = st.slider("Next over: runs scored", 0, 24, 8)
    whatif_wkt  = st.radio("Next over: wicket?", ["No", "Yes"], horizontal=True)

# ══ RIGHT: Results ══════════════════════════════════════════════════════════════
with right:
    if predict_btn or True:   # show results on load with defaults too
        prob = predict_win_probability(
            batting_team, bowling_team, target, score, balls_played, wickets
        )
        bowl_prob = 1 - prob
        bat_pct  = round(prob * 100, 1)
        bowl_pct = round(bowl_prob * 100, 1)

        confidence, conf_cls = get_confidence(prob)
        momentum, mom_cls    = get_momentum(prob, rrr, crr, 10 - wickets)
        insight = get_insight(batting_team, prob, rrr, crr, 10 - wickets, balls_left)
        insight_cls = "safe" if prob > 0.6 else ("danger" if prob < 0.4 else "")

        # Colour: gold for batting, blue for bowling
        bat_color  = "#f0c040"
        bowl_color = "#3b82f6"
        prob_color = bat_color if prob > 0.5 else bowl_color

        # ── Big probability card ──
        st.markdown(f"""
        <div class="prob-display">
            <div class="prob-number" style="color:{prob_color}">{bat_pct}%</div>
            <div class="prob-team">{batting_team}</div>
            <div class="prob-label">Win Probability</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Split bar ──
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;margin-top:1rem;font-size:0.8rem;color:#9ca3af;">
            <span>🏏 {batting_team} &nbsp;{bat_pct}%</span>
            <span>{bowl_pct}% &nbsp;{bowling_team} 🎯</span>
        </div>
        <div class="split-bar-wrap">
            <div class="split-bar-left" style="width:{bat_pct}%"></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Badges ──
        st.markdown(f"""
        <div style="display:flex;gap:8px;margin:0.8rem 0;flex-wrap:wrap;">
            <span class="momentum-badge {conf_cls}">🎯 Confidence: {confidence}</span>
            <span class="momentum-badge {mom_cls}">⚡ Momentum: {momentum}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Insight ──
        st.markdown(f'<div class="insight-box {insight_cls}">{insight}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── What-if result ──
        new_score    = score + whatif_runs
        new_wickets  = min(wickets + (1 if whatif_wkt == "Yes" else 0), 10)
        new_balls    = balls_played + 6
        whatif_prob  = predict_win_probability(
            batting_team, bowling_team, target, new_score, new_balls, new_wickets
        )
        delta = (whatif_prob - prob) * 100
        delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
        delta_color = "#22c55e" if delta >= 0 else "#ef4444"

        st.markdown(f"""
        <div class="card-title">What-If: Next Over (+{whatif_runs} runs{', 1 wkt' if whatif_wkt=='Yes' else ''})</div>
        <div style="display:flex;align-items:center;gap:1.5rem;padding:1rem;background:#13182b;border-radius:12px;border:1px solid rgba(255,255,255,0.05);">
            <div>
                <div style="color:#6b7280;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Updated Probability</div>
                <div style="font-family:'Bebas Neue',sans-serif;font-size:2.5rem;color:{bat_color}">{round(whatif_prob*100,1)}%</div>
            </div>
            <div>
                <div style="color:#6b7280;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px;">Change</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:700;color:{delta_color}">{delta_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Charts ──
        chart_left, chart_right = st.columns(2)

        with chart_left:
            # Win probability curve
            scores_x, probs_y = compute_probability_curve(
                batting_team, bowling_team, target, balls_played, wickets
            )
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=scores_x, y=probs_y,
                mode="lines",
                line=dict(color="#f0c040", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(240,192,64,0.08)",
                hovertemplate="Score: %{x}<br>Win Prob: %{y:.1f}%<extra></extra>",
            ))
            fig_curve.add_vline(x=score, line_dash="dash", line_color="#ff6b35", line_width=1.5)
            fig_curve.add_annotation(
                x=score, y=bat_pct, text=f" Now ({bat_pct}%)",
                showarrow=False, font=dict(color="#ff6b35", size=11), xanchor="left"
            )
            fig_curve.update_layout(
                title=dict(text="Win Probability vs Score", font=dict(color="#9ca3af", size=12)),
                xaxis=dict(title="Score", color="#6b7280", gridcolor="#1e2640", showgrid=True),
                yaxis=dict(title="Win %", color="#6b7280", gridcolor="#1e2640", range=[0, 100]),
                paper_bgcolor="#13182b", plot_bgcolor="#13182b",
                font=dict(color="#9ca3af"),
                margin=dict(l=10, r=10, t=40, b=10),
                height=260,
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        with chart_right:
            # Feature importance
            if importance_df is not None:
                top_n = importance_df.head(8)
                # Shorten feature names for display
                display_names = (
                    top_n["feature"]
                    .str.replace("batting_team_", "BAT: ", regex=False)
                    .str.replace("bowling_team_", "BOWL: ", regex=False)
                    .str.replace("_", " ", regex=False)
                    .str.title()
                )
                fig_imp = go.Figure(go.Bar(
                    x=top_n["importance"].values[::-1],
                    y=display_names.values[::-1],
                    orientation="h",
                    marker=dict(
                        color=top_n["importance"].values[::-1],
                        colorscale=[[0, "#1e2640"], [1, "#f0c040"]],
                    ),
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                ))
                fig_imp.update_layout(
                    title=dict(text="Feature Importance", font=dict(color="#9ca3af", size=12)),
                    xaxis=dict(color="#6b7280", gridcolor="#1e2640"),
                    yaxis=dict(color="#9ca3af"),
                    paper_bgcolor="#13182b", plot_bgcolor="#13182b",
                    font=dict(color="#9ca3af"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=260,
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")

        # ── Model disclaimer ──
        st.markdown("""
        <div style="margin-top:1rem;padding:0.8rem 1rem;background:#13182b;border-radius:10px;
                    border:1px solid rgba(255,255,255,0.05);font-size:0.75rem;color:#6b7280;line-height:1.6;">
            <b style="color:#9ca3af;">⚠️ Model Note</b> — Trained on synthetic IPL-style data (~44K snapshots).
            Probabilities are illustrative. Real-world accuracy depends on live pitch conditions, player form,
            and match context not captured here. ROC-AUC on held-out test set: <b style="color:#f0c040;">0.9888</b>.
        </div>
        """, unsafe_allow_html=True)