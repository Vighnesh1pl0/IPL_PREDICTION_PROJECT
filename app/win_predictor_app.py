import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="IPL Win Probability",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* MAIN BACKGROUND */
body {
    background: linear-gradient(180deg, #0b0f19, #020617);
    color: #e5e7eb;
}

/* SIDEBAR – ADVANCED COLOR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b, #312e81);
    color: #e5e7eb;
    border-right: 1px solid rgba(255,255,255,0.15);
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #e5e7eb !important;
}

/* HEADINGS */
h1, h2, h3 {
    color: #e5e7eb;
    font-weight: 700;
}

/* CARDS */
.card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 12px 45px rgba(0,0,0,0.6);
}

/* METRIC BOX */
.metric-box {
    background: linear-gradient(135deg, #0f172a, #020617);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    border: 1px solid rgba(99,102,241,0.35);
    box-shadow: 0 0 22px rgba(99,102,241,0.25);
}

.metric-box h1 {
    color: #60a5fa;
    font-size: 42px;
}

.metric-box h3 {
    color: #c7d2fe;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #22d3ee);
    color: #020617;
    font-weight: 700;
    border-radius: 14px;
    height: 3em;
    border: none;
    box-shadow: 0 0 24px rgba(99,102,241,0.55);
}

.stButton>button:hover {
    transform: scale(1.04);
}

/* FOOTER */
.footer {
    color: #94a3b8;
    text-align: center;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align:center;
background: linear-gradient(90deg,#60a5fa,#a78bfa);
-webkit-background-clip: text;
color: transparent;
font-size: 52px;">
IPL Win Probability Dashboard
</h1>
<p style="text-align:center;color:#c7d2fe;">
Advanced ML-based second innings analysis
</p>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    model = joblib.load("../models/best_winprob_model.pkl")
    columns = joblib.load("../models/x_columns.pkl")
    return model, columns

model, x_columns = load_model()

st.sidebar.header("Match Setup")

teams = sorted({c.split("_")[1] for c in x_columns if c.startswith("team1_")})

batting_team = st.sidebar.selectbox("Batting Team", teams)
bowling_team = st.sidebar.selectbox("Bowling Team", teams)

st.sidebar.markdown("First Innings Details")

inn1_total = st.sidebar.slider("Total Runs", 50, 300, 160)
pp_runs = st.sidebar.slider("Powerplay Runs", 0, 120, 40)
mid_runs = st.sidebar.slider("Middle Overs Runs", 0, 150, 75)
death_runs = st.sidebar.slider("Death Overs Runs", 0, 100, 45)

pp_wk = st.sidebar.slider("Powerplay Wickets", 0, 5, 1)
mid_wk = st.sidebar.slider("Middle Overs Wickets", 0, 5, 2)
death_wk = st.sidebar.slider("Death Overs Wickets", 0, 5, 1)
extras = st.sidebar.slider("Extras", 0, 20, 4)

data = pd.DataFrame([[
    inn1_total, pp_runs, mid_runs, death_runs,
    pp_wk, mid_wk, death_wk, extras
]], columns=[
    "inn1_total", "pp_runs", "mid_runs", "death_runs",
    "pp_wickets", "mid_wickets", "death_wickets", "extras"
])

for col in x_columns:
    if col.startswith("team1_"):
        data[col] = 1 if col == f"team1_{batting_team}" else 0
    elif col.startswith("team2_"):
        data[col] = 1 if col == f"team2_{bowling_team}" else 0
    elif col not in data.columns:
        data[col] = 0

data = data[x_columns]

tab1, tab2, tab3 = st.tabs(["Overview", "Prediction", "Insights"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-box'><h3>Total Runs</h3><h1>{inn1_total}</h1></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><h3>Wickets Lost</h3><h1>{pp_wk + mid_wk + death_wk}</h1></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-box'><h3>Run Rate</h3><h1>{inn1_total/20:.2f}</h1></div>", unsafe_allow_html=True)

    phase_df = pd.DataFrame({
        "Phase": ["Powerplay", "Middle", "Death"],
        "Runs": [pp_runs, mid_runs, death_runs]
    })

    fig = px.bar(phase_df, x="Phase", y="Runs", text="Runs", template="plotly_dark")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    prob = model.predict_proba(data)[0][1]

    c1, c2 = st.columns([2, 1])
    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": f"{batting_team} win probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22c55e" if prob > 0.5 else "#ef4444"},
                "steps": [
                    {"range": [0, 50], "color": "#7f1d1d"},
                    {"range": [50, 100], "color": "#14532d"}
                ]
            }
        ))
        gauge.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(gauge, use_container_width=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if prob > 0.65:
            st.success("Strong batting advantage")
        elif prob > 0.45:
            st.warning("Match evenly poised")
        else:
            st.error("Bowling side on top")
        st.markdown(f"Confidence: `{prob*100:.1f}%`")
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    insight_df = pd.DataFrame({
        "Factor": ["Powerplay impact", "Middle overs stability", "Death overs acceleration", "Extras conceded"],
        "Impact": [
            pp_runs - pp_wk * 10,
            mid_runs - mid_wk * 8,
            death_runs - death_wk * 12,
            -extras
        ]
    })

    fig = px.bar(insight_df, x="Impact", y="Factor", orientation="h", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>IPL Win Predictor • Machine Learning Project</p>", unsafe_allow_html=True)
