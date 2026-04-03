import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import json
import mlflow

PUMA_RED = "#e41e26"
WHITE = "#ffffff"
GRAY = "#888888"
# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="⚡ Smart Campus Energy Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS – DARK THEME ====================
st.markdown("""
<style>

/* FONT */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* MAIN BACKGROUND */
.stApp {
    background-color: #000000;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #0a0a0a;
    border-right: 1px solid #1a1a1a;
}

/* HEADINGS */
h1 {
    color: #ffffff;
    font-weight: 900;
    font-size: 2.8rem;
    letter-spacing: -1px;
}

h2 {
    color: #ffffff;
    font-weight: 700;
    font-size: 1.8rem;
    border-bottom: 1px solid #222;
    padding-bottom: 10px;
}

h3 {
    color: #e0e0e0;
    font-weight: 600;
}

/* TEXT */
p, label, span {
    color: #cfcfcf;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background: #0f0f0f;
    border: 1px solid #1f1f1f;
    border-radius: 16px;
    padding: 24px;
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    border: 1px solid #e41e26;
    transform: translateY(-4px);
}

div[data-testid="stMetricValue"] {
    color: #ffffff;
    font-weight: 700;
    font-size: 2rem;
}

/* BUTTONS */
.stButton > button {
    background-color: #e41e26;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 26px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #ff2c35;
}

/* SELECT BOX */
.stSelectbox div {
    background-color: #0f0f0f;
    color: white;
    border: 1px solid #222;
    border-radius: 8px;
}

/* RADIO */
.stRadio > div {
    background: #0f0f0f;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #1a1a1a;
}

/* SLIDER */
.stSlider > div {
    color: white;
}

/* DATAFRAME */
.stDataFrame {
    background-color: #0f0f0f;
    border-radius: 16px;
    border: 1px solid #222;
}

/* EXPANDER */
.streamlit-expanderHeader {
    background-color: #0f0f0f;
    border-radius: 10px;
    border: 1px solid #1f1f1f;
}

/* SECTION CONTAINER */
.section-container {
    background: #0f0f0f;
    border: 1px solid #1f1f1f;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
}

/* FOOTER */
.footer {
    border-top: 1px solid #222;
    padding: 20px;
    color: #777;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #e41e26;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def get_drift_status():
    """Read drift report and return status message and CSS class."""
    report_path = 'drift_report.json'
    if not os.path.exists(report_path):
        return None, "⚠️ No drift report available. Run drift detection first.", "drift-info"
    try:
        with open(report_path) as f:
            report = json.load(f)
        last = report.get('last_check', {})
        if last.get('drift_detected', False):
            details = last.get('details', {})
            drifted = [k for k, v in details.items() if v.get('drift_detected', False)]
            msg = f"🚨 **Drift Detected!** Features: {', '.join(drifted)}. Consider retraining."
            return True, msg, "drift-error"
        else:
            return False, "✅ No significant drift detected.", "drift-success"
    except Exception as e:
        return None, f"⚠️ Error reading drift report: {e}", "drift-warning"


@st.cache_resource
def load_production_model():
    """Load the production model from MLflow."""
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        model = mlflow.pyfunc.load_model("models:/EnergyPredictor1/Production")
        return model
    except Exception:
        return None


def simulate_fallback(building, hour, temp, occupancy, weekend, event):
    """Fallback prediction when ML model is unavailable."""
    base = 100 + hour * 8
    factors = {
        'Library': 1.2,
        'Computer Lab': 1.5,
        'Classroom': 1.0,
        'Hostel': 0.8,
        'Auditorium': 1.8
    }
    base *= factors.get(building, 1.0)
    if temp > 28:
        base *= 1.3
    if occupancy > 0.7:
        base *= 1.2
    if event:
        base *= 1.4
    if weekend:
        base *= 0.7
    return round(base, 2)


def predict_energy(building, hour, temp, occupancy, weekend, event):
    """Predict energy consumption using ML model or fallback."""
    model = load_production_model()
    if model is not None:
        features = pd.DataFrame([{
            'hour_of_day': hour,
            'temperature': temp,
            'occupancy': occupancy
        }])
        try:
            pred = model.predict(features)[0]
            factors = {
                'Library': 1.2,
                'Computer Lab': 1.5,
                'Classroom': 1.0,
                'Hostel': 0.8,
                'Auditorium': 1.8
            }
            pred *= factors.get(building, 1.0)
            return round(pred, 2)
        except Exception:
            return simulate_fallback(building, hour, temp, occupancy, weekend, event)
    else:
        return simulate_fallback(building, hour, temp, occupancy, weekend, event)


@st.cache_data
def load_datasets():
    """Load all available datasets from the raw data folder."""
    files = {
        "Research Campus Energy": "data/raw/research_based_campus_energy.csv",
        "OpenEI Building Energy": "data/raw/openei_building_energy.csv",
        "UCI Appliance Energy": "data/raw/uci_energy_data.csv",
        "Research Citations": "data/raw/research_citations.csv"
    }
    datasets = {}
    for name, path in files.items():
        if os.path.exists(path):
            try:
                datasets[name] = pd.read_csv(path)
            except Exception:
                pass  # ignore files that can't be read
    return datasets


# ==================== MAIN APP ====================

st.markdown("""
<h1>SMART CAMPUS ENERGY</h1>
<p style="color:#777; font-size:18px; margin-top:-10px;">
AI-Powered Energy Optimization Platform
</p>
""", unsafe_allow_html=True)
# Load datasets
datasets = load_datasets()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🌌 Data Explorer")
    if datasets:
        selected = st.selectbox("📁 Choose Dataset", list(datasets.keys()))
        df = datasets[selected].copy()

        st.markdown(f"""
        <div style="background:#1e2632; border-left:4px solid #4f9cf7; padding:12px; border-radius:8px; margin:10px 0;">
            <b style="color:#e0e8f0;">{selected}</b><br>
            <span style="color:#9aa8b9;">Records: {len(df):,} | Columns: {len(df.columns)}</span>
        </div>
        """, unsafe_allow_html=True)

        # Date range filter if timestamp column exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            min_d, max_d = df['timestamp'].min().date(), df['timestamp'].max().date()
            dr = st.date_input("📅 Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)
            if len(dr) == 2:
                df = df[(df['timestamp'].dt.date >= dr[0]) & (df['timestamp'].dt.date <= dr[1])]
    else:
        st.error("❌ No datasets found.")
        st.stop()

    # Drift status
    drift_detected, drift_msg, drift_class = get_drift_status()
    st.markdown(f'<div class="{drift_class}">{drift_msg}</div>', unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio(
        "🔮 Analysis Mode",
        [
            "📈 Energy Patterns",
            "🏢 Building Comparison",
            "💰 Savings Analysis",
            "🌱 Environmental Impact",
            "⚡ Live Prediction"
        ]
    )

# ==================== UTILITY ====================
# Identify energy-related columns
energy_cols = [
    c for c in df.columns
    if 'energy' in c.lower() or 'kwh' in c.lower() or 'meter' in c.lower()
]
if not energy_cols:
    # Fallback: use first numeric column
    energy_cols = df.select_dtypes(include=[np.number]).columns[:1]

# ==================== DATASET PREVIEW ====================
with st.expander("🔍 Dataset Preview", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    cols = st.columns(3)
    cols[0].metric("Records", len(df))
    cols[1].metric("Columns", len(df.columns))
    cols[2].metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))

# ==================== MAIN CONTENT SECTIONS ====================

if mode == "📈 Energy Patterns":
    st.markdown("<h2>📈 Energy Consumption Patterns</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        energy_col = st.selectbox("Energy Column", energy_cols, key="pat_eng")

        # Daily pattern
        if 'hour_of_day' in df.columns or 'hour' in df.columns:
            hour_col = 'hour_of_day' if 'hour_of_day' in df.columns else 'hour'
            daily = df.groupby(hour_col)[energy_col].mean().reset_index()
            fig = px.line(
                daily,
                x=hour_col,
                y=energy_col,
                title="Daily Pattern",
                markers=True,
                color_discrete_sequence=[PUMA_RED]
            )
            fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="white", family="Inter"),
    title_font=dict(size=20, color="white")
)
            st.plotly_chart(fig, use_container_width=True)

        # Weekly pattern
        if 'day_of_week' in df.columns:
            weekly = df.groupby('day_of_week')[energy_col].mean().reset_index()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig = px.bar(
                weekly,
                x='day_of_week',
                y=energy_col,
                title="Weekly Pattern",
                color=energy_col,
                color_continuous_scale='Blues'
            )
            fig.update_xaxes(ticktext=days, tickvals=list(range(7)))
            fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="white", family="Inter"),
    title_font=dict(size=20, color="white")
)

    with col2:
        fig = px.histogram(
            df,
            x=energy_cols[0],
            title="Energy Distribution",
            nbins=50,
            color_discrete_sequence=[PUMA_RED]
        )
        fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="white", family="Inter"),
    title_font=dict(size=20, color="white")
)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[energy_cols[0]].describe().round(2), use_container_width=True)

elif mode == "🏢 Building Comparison":
    st.markdown("<h2>🏢 Building Energy Comparison</h2>", unsafe_allow_html=True)

    building_cols = [c for c in df.columns if 'building' in c.lower() or 'site' in c.lower()]
    if building_cols and energy_cols:
        col1, col2 = st.columns(2)

        with col1:
            building_col = st.selectbox("Building Column", building_cols)
            energy_col = st.selectbox("Energy Column", energy_cols, key="build_eng")

            stats = df.groupby(building_col)[energy_col].agg(['mean', 'std', 'count']).round(2)
            stats = stats.sort_values('mean', ascending=False)

            fig = px.bar(
                stats.reset_index(),
                x=building_col,
                y='mean',
                error_y='std',
                title="Average Energy by Building",
                color='mean',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="white", family="Inter"),
    title_font=dict(size=20, color="white")
)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(stats.head(10), use_container_width=True)
            avg = df[energy_col].mean()
            st.metric("Overall Average", f"{avg:.1f} kWh")
            st.metric("Potential Savings (15%)", f"{avg * 0.15:.1f} kWh/day")
    else:
        st.warning("Building or energy column not found.")

elif mode == "💰 Savings Analysis":
    st.markdown("<h2>💰 Cost & Savings Analysis</h2>", unsafe_allow_html=True)

    if len(energy_cols) > 0:
        energy_col = energy_cols[0]
        avg = df[energy_col].mean()
        percent = st.slider("Optimization %", 5, 25, 15, 1) / 100
        current = avg
        optimized = current * (1 - percent)
        saved = current - optimized
        rate = st.number_input("Electricity Rate (₹/kWh)", 5.0, 15.0, 8.0, 0.5)
        cost_saved = saved * rate
        co2_saved = saved * 0.82

        cols = st.columns(4)
        cols[0].metric("Current", f"{current:.1f} kWh/day")
        cols[1].metric("Optimized", f"{optimized:.1f} kWh/day", delta=f"-{percent * 100:.0f}%")
        cols[2].metric("Energy Saved", f"{saved:.1f} kWh/day")
        cols[3].metric("Daily Savings", f"₹{cost_saved:.0f}")

        data = pd.DataFrame({
            'Period': ['Daily', 'Monthly', 'Annual'],
            'Energy Saved (kWh)': [saved, saved * 30, saved * 365],
            'Cost Saved (₹)': [cost_saved, cost_saved * 30, cost_saved * 365],
            'CO₂ Reduced (kg)': [co2_saved, co2_saved * 30, co2_saved * 365]
        })
        st.dataframe(data, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Current',
            x=['Usage'],
            y=[current],
            marker_color='#e07b7b',
            text=[f"{current:.1f}"]
        ))
        fig.add_trace(go.Bar(
            name='Optimized',
            x=['Usage'],
            y=[optimized],
            marker_color='#4f9cf7',
            text=[f"{optimized:.1f}"]
        ))
        fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="white", family="Inter"),
    title_font=dict(size=20, color="white")
)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No energy column.")

elif mode == "🌱 Environmental Impact":
    st.markdown("<h2>🌱 Environmental Impact</h2>", unsafe_allow_html=True)

    if energy_cols:
        energy_col = energy_cols[0]
        total = df[energy_col].sum()
        co2_total = total * 0.82
        trees = co2_total / 20
        cars = co2_total / 4200

        cols = st.columns(4)
        cols[0].metric("Total Energy", f"{total:,.0f} kWh")
        cols[1].metric("CO₂ Emissions", f"{co2_total:,.0f} kg")
        cols[2].metric("Trees Needed", f"{trees:.0f}")
        cols[3].metric("Car Equivalent", f"{cars:.1f} years")

        env_df = pd.DataFrame({
            'Scenario': ['Current', 'With 15% Optimization'],
            'CO₂ (tons)': [co2_total / 1000, co2_total * 0.85 / 1000]
        })
        fig = px.bar(
            env_df,
            x='Scenario',
            y='CO₂ (tons)',
            text='CO₂ (tons)',
            color='Scenario',
            color_discrete_sequence=[PUMA_RED]
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="white", family="Inter"),
    title_font=dict(size=20, color="white")
)
        st.plotly_chart(fig, use_container_width=True)

        # Show research citations if available
        if os.path.exists("data/raw/research_citations.csv"):
            with st.expander("📚 Research Basis"):
                citations = pd.read_csv("data/raw/research_citations.csv")
                for _, row in citations.iterrows():
                    st.markdown(f"**{row['study']}** ({row['year']}) – {row['key_finding']}")
    else:
        st.warning("No energy column.")

elif mode == "⚡ Live Prediction":
    st.markdown("<h2>⚡ Live Energy Prediction</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        building = st.selectbox(
            "Building",
            ["Library", "Computer Lab", "Classroom", "Hostel", "Auditorium"]
        )
        hour = st.slider("Hour", 0, 23, 14)
        temp = st.slider("Temperature (°C)", 0, 45, 32)

    with col2:
        occ = st.slider("Occupancy %", 0, 100, 75) / 100
        weekend = st.checkbox("Weekend")
        event = st.checkbox("Special Event")

    if st.button("🔮 Predict Energy", type="primary"):
        with st.spinner("Calculating..."):
            pred = predict_energy(building, hour, temp, occ, weekend, event)
            opt = pred * 0.85
            saved = pred - opt
            cost = saved * 8
            co2 = saved * 0.82

            cols = st.columns(4)
            cols[0].metric("Predicted", f"{pred:.1f} kWh")
            cols[1].metric("Optimized", f"{opt:.1f} kWh", delta=f"-{saved:.1f}")
            cols[2].metric("Cost Savings", f"₹{cost:.0f}/day")
            cols[3].metric("CO₂ Reduction", f"{co2:.1f} kg/day")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                title={'text': "Predicted Energy (kWh)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, pred * 1.5]},
                    'bar': {'color': '#4f9cf7'},
                    'steps': [
                        {'range': [0, opt], 'color': '#2e6b4e'},
                        {'range': [opt, pred], 'color': '#a55e5e'}
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 4},
                        'thickness': 0.75,
                        'value': pred
                    }
                }
            ))
            fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(color="white", family="Inter"),
    title_font=dict(size=20, color="white")
)
            st.plotly_chart(fig, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <b>⚡ Smart Campus Energy Optimizer</b> – AI‑Powered MLOps Pipeline | Data: ASHRAE, UCI, OpenEI
</div>
""", unsafe_allow_html=True)