# dashboard/app.py - COMPLETE VERSION WITH LIVE PREDICTION
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import mlflow

def load_drift_status():
    """Load latest drift report and return warning message if drift detected."""
    report_path = 'drift_report.json'
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path) as f:
            report = json.load(f)
        last = report.get('last_check', {})
        if last.get('drift_detected', False):
            details = last.get('details', {})
            drifted = [k for k, v in details.items() if v.get('drift_detected', False)]
            msg = f"🚨 **Drift Detected!** Features drifted: {', '.join(drifted)}. Consider retraining."
            return msg
        else:
            return None
    except Exception as e:
        print(f"Could not load drift report: {e}")
        return None
    

# ==================== MLFLOW MODEL LOADING ====================
@st.cache_resource
def load_production_model():
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        model = mlflow.pyfunc.load_model("models:/EnergyPredictor1/Production")
        st.success("✅ Loaded production model from MLflow")
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load production model: {e}. Using fallback simulation.")
        return None

model = load_production_model()

# ==================== FALLBACK SIMULATION ====================
def simulate_fallback(building, hour, temp, occupancy, weekend, event):
    """Rule-based fallback when MLflow model is unavailable."""
    base = 100 + hour * 8
    building_factors = {
        'Library': 1.2, 'Computer Lab': 1.5, 'Classroom': 1.0,
        'Hostel': 0.8, 'Auditorium': 1.8
    }
    base *= building_factors.get(building, 1.0)
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
    """Use MLflow production model if available, else fallback."""
    if model is not None:
        # Prepare features – adjust column names to match your training data
        features = pd.DataFrame([{
            'hour_of_day': hour,
            'temperature': temp,
            'occupancy': occupancy,
            # If your model also used 'is_weekend' or 'special_event', uncomment:
            # 'is_weekend': int(weekend),
            # 'special_event': int(event)
        }])
        try:
            pred = model.predict(features)[0]
            # Apply building factor (since model wasn't trained with building type)
            building_factors = {
                'Library': 1.2, 'Computer Lab': 1.5, 'Classroom': 1.0,
                'Hostel': 0.8, 'Auditorium': 1.8
            }
            pred *= building_factors.get(building, 1.0)
            return round(pred, 2)
        except Exception as e:
            st.error(f"Prediction error: {e}. Using fallback.")
            return simulate_fallback(building, hour, temp, occupancy, weekend, event)
    else:
        return simulate_fallback(building, hour, temp, occupancy, weekend, event)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Smart Campus Energy Optimizer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Campus Energy Optimizer")
st.markdown("### Real Dataset Analysis Dashboard")

# ==================== LOAD REAL DATASETS ====================
@st.cache_data
def load_real_datasets():
    datasets = {}
    data_files = {
        "Research Campus Energy": "data/raw/research_based_campus_energy.csv",
        "OpenEI Building Energy": "data/raw/openei_building_energy.csv", 
        "UCI Appliance Energy": "data/raw/uci_energy_data.csv",
        "Research Citations": "data/raw/research_citations.csv"
    }
    for name, path in data_files.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                datasets[name] = df
                st.sidebar.success(f"✅ Loaded: {name}")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error loading {name}: {e}")
        else:
            st.sidebar.error(f"❌ Not found: {name}")
    return datasets

# ==================== SIDEBAR ====================

# Display drift warning in sidebar
drift_warning = load_drift_status()
if drift_warning:
    st.sidebar.warning(drift_warning)

    
with st.sidebar:
    st.header("📊 Data Selection")
    
    with st.spinner("Loading real datasets..."):
        datasets = load_real_datasets()
    
    if datasets:
        selected_dataset = st.selectbox(
            "Choose Dataset",
            list(datasets.keys())
        )
        df = datasets[selected_dataset]
        
        st.info(f"""
        **Dataset:** {selected_dataset}
        **Records:** {len(df):,}
        **Columns:** {len(df.columns)}
        **Size:** {df.memory_usage(deep=True).sum()/(1024*1024):.1f} MB
        """)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                mask = (df['timestamp'].dt.date >= date_range[0]) & \
                       (df['timestamp'].dt.date <= date_range[1])
                df = df[mask]
    
    st.markdown("---")
    st.header("📈 Analysis Type")
    analysis_type = st.radio(
        "Select Analysis",
        ["📈 Energy Patterns", "🏢 Building Comparison", "💰 Savings Analysis", 
         "🌱 Environmental Impact", "🔮 Live Prediction"]
    )

# ==================== MAIN DASHBOARD ====================
if not datasets:
    st.error("❌ No datasets found in data/raw/ folder!")
    st.info("""
    Please ensure you have these files:
    - `research_based_campus_energy.csv`
    - `openei_building_energy.csv`
    - `uci_energy_data.csv`
    
    Run `python main.py --real` first to download datasets.
    """)
else:
    with st.expander("🔍 Dataset Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))
    
    # -------------------- Energy Patterns --------------------
    if analysis_type == "📈 Energy Patterns":
        st.header("📈 Energy Consumption Patterns")
        col1, col2 = st.columns(2)
        with col1:
            energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'kwh' in col.lower() or 'meter' in col.lower()]
            if energy_cols:
                energy_col = st.selectbox("Select Energy Column", energy_cols)
                if 'hour_of_day' in df.columns or 'hour' in df.columns:
                    hour_col = 'hour_of_day' if 'hour_of_day' in df.columns else 'hour'
                    daily_pattern = df.groupby(hour_col)[energy_col].mean().reset_index()
                    fig1 = px.line(daily_pattern, x=hour_col, y=energy_col, 
                                  title="📅 Average Daily Energy Pattern",
                                  markers=True)
                    fig1.update_layout(xaxis_title="Hour of Day", yaxis_title="Energy (kWh)")
                    st.plotly_chart(fig1, use_container_width=True)
                if 'day_of_week' in df.columns:
                    weekly_pattern = df.groupby('day_of_week')[energy_col].mean().reset_index()
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    fig2 = px.bar(weekly_pattern, x='day_of_week', y=energy_col,
                                 title="📆 Weekly Energy Pattern",
                                 labels={'day_of_week': 'Day', energy_col: 'Energy (kWh)'})
                    fig2.update_xaxes(ticktext=days, tickvals=list(range(7)))
                    st.plotly_chart(fig2, use_container_width=True)
        with col2:
            if energy_cols:
                fig3 = px.histogram(df, x=energy_cols[0], 
                                   title="📊 Energy Distribution",
                                   nbins=50)
                fig3.update_layout(xaxis_title="Energy (kWh)", yaxis_title="Frequency")
                st.plotly_chart(fig3, use_container_width=True)
                st.subheader("📊 Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max', '25%', '75%'],
                    'Value': [
                        f"{df[energy_cols[0]].mean():.1f} kWh",
                        f"{df[energy_cols[0]].std():.1f} kWh",
                        f"{df[energy_cols[0]].min():.1f} kWh",
                        f"{df[energy_cols[0]].max():.1f} kWh",
                        f"{df[energy_cols[0]].quantile(0.25):.1f} kWh",
                        f"{df[energy_cols[0]].quantile(0.75):.1f} kWh"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # -------------------- Building Comparison --------------------
    elif analysis_type == "🏢 Building Comparison":
        st.header("🏢 Building Energy Comparison")
        building_cols = [col for col in df.columns if 'building' in col.lower() or 'site' in col.lower()]
        energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'kwh' in col.lower()]
        if building_cols and energy_cols:
            building_col = st.selectbox("Select Building Column", building_cols)
            energy_col = st.selectbox("Select Energy Column", energy_cols)
            building_stats = df.groupby(building_col)[energy_col].agg(['mean', 'std', 'count']).round(2)
            building_stats = building_stats.sort_values('mean', ascending=False)
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(building_stats.reset_index(), 
                            x=building_col, y='mean',
                            title="🏢 Average Energy by Building",
                            error_y='std')
                fig1.update_layout(xaxis_title="Building", yaxis_title="Average Energy (kWh)")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.subheader("📊 Building Statistics")
                st.dataframe(building_stats.head(10), use_container_width=True)
                avg_energy = df[energy_col].mean()
                potential_savings = avg_energy * 0.15
                daily_cost_savings = potential_savings * 8
                monthly_savings = daily_cost_savings * 30
                st.metric("Average Energy", f"{avg_energy:.1f} kWh")
                st.metric("Potential Savings (15%)", f"{potential_savings:.1f} kWh/day")
                st.metric("Cost Savings", f"₹{daily_cost_savings:.0f}/day")
                st.metric("Monthly Impact", f"₹{monthly_savings:,.0f}")
    
    # -------------------- Savings Analysis --------------------
    elif analysis_type == "💰 Savings Analysis":
        st.header("💰 Cost & Savings Analysis")
        energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'kwh' in col.lower()]
        if energy_cols:
            energy_col = energy_cols[0]
            avg_energy = df[energy_col].mean()
            savings_percent = st.slider("Optimization Percentage", 5, 25, 15) / 100
            current_daily = avg_energy
            optimized_daily = current_daily * (1 - savings_percent)
            energy_saved = current_daily - optimized_daily
            electricity_rate = st.number_input("Electricity Rate (₹/kWh)", 5.0, 15.0, 8.0, 0.5)
            daily_cost_saved = energy_saved * electricity_rate
            monthly_cost_saved = daily_cost_saved * 30
            annual_cost_saved = daily_cost_saved * 365
            co2_per_kwh = st.number_input("CO₂ per kWh (kg)", 0.5, 1.5, 0.82, 0.01)
            daily_co2_saved = energy_saved * co2_per_kwh
            annual_co2_saved = daily_co2_saved * 365
            trees_equivalent = annual_co2_saved / 20
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Usage", f"{current_daily:.1f} kWh/day")
            with col2:
                st.metric("Optimized Usage", f"{optimized_daily:.1f} kWh/day", 
                         delta=f"-{savings_percent*100:.1f}%")
            with col3:
                st.metric("Energy Saved", f"{energy_saved:.1f} kWh/day")
            with col4:
                st.metric("Daily Savings", f"₹{daily_cost_saved:.0f}")
            breakdown_data = {
                'Period': ['Daily', 'Monthly', 'Annual'],
                'Energy Saved (kWh)': [energy_saved, energy_saved*30, energy_saved*365],
                'Cost Savings (₹)': [daily_cost_saved, monthly_cost_saved, annual_cost_saved],
                'CO₂ Reduced (kg)': [daily_co2_saved, daily_co2_saved*30, annual_co2_saved]
            }
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current',
                x=['Energy Usage'],
                y=[current_daily],
                marker_color='red',
                text=[f"{current_daily:.1f} kWh"],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                name='Optimized',
                x=['Energy Usage'],
                y=[optimized_daily],
                marker_color='green',
                text=[f"{optimized_daily:.1f} kWh"],
                textposition='auto'
            ))
            fig.update_layout(title='Energy Optimization Impact', yaxis_title='Energy (kWh/day)', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    # -------------------- Environmental Impact --------------------
    elif analysis_type == "🌱 Environmental Impact":
        st.header("🌱 Environmental Impact Analysis")
        energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'kwh' in col.lower()]
        if energy_cols:
            energy_col = energy_cols[0]
            total_energy = df[energy_col].sum()
            co2_per_kwh = 0.82
            total_co2 = total_energy * co2_per_kwh
            trees_needed = total_co2 / 20
            cars_equivalent = total_co2 / 4200
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Energy", f"{total_energy:,.0f} kWh")
            with col2:
                st.metric("CO₂ Emissions", f"{total_co2:,.0f} kg")
            with col3:
                st.metric("Trees Needed", f"{trees_needed:.0f} trees")
            with col4:
                st.metric("Car Equivalents", f"{cars_equivalent:.1f} cars/year")
            env_data = {
                'Source': ['Current Campus', 'With 15% Optimization'],
                'CO₂ Emissions (tons)': [total_co2/1000, total_co2*0.85/1000],
                'Energy (MWh)': [total_energy/1000, total_energy*0.85/1000]
            }
            env_df = pd.DataFrame(env_data)
            fig = px.bar(env_df, x='Source', y='CO₂ Emissions (tons)',
                        title='Carbon Emissions Comparison',
                        text='CO₂ Emissions (tons)')
            fig.update_traces(texttemplate='%{text:.1f} tons', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            citations_path = "data/raw/research_citations.csv"
            if os.path.exists(citations_path):
                st.subheader("📚 Research Basis")
                citations = pd.read_csv(citations_path)
                for _, row in citations.iterrows():
                    with st.expander(f"📄 {row['study']} ({row['year']})"):
                        st.write(f"**Finding:** {row['key_finding']}")
                        st.write(f"**Source:** {row['source']}")
    
    # -------------------- Live Prediction --------------------
    elif analysis_type == "🔮 Live Prediction":
        st.header("🔮 Live Energy Prediction")
        col1, col2 = st.columns(2)
        with col1:
            building = st.selectbox(
                "Building Type",
                ["Library", "Computer Lab", "Classroom", "Hostel", "Auditorium"]
            )
            hour = st.slider("Hour of Day", 0, 23, 14)
            temperature = st.slider("Temperature (°C)", 0, 45, 32)
        with col2:
            occupancy = st.slider("Occupancy (%)", 0, 100, 75) / 100
            weekend = st.checkbox("Weekend")
            event = st.checkbox("Special Event (Exam/Holiday)")
        
        if st.button("Predict Energy"):
            with st.spinner("Calculating..."):
                predicted = predict_energy(building, hour, temperature, occupancy, weekend, event)
                optimized = predicted * 0.85
                saved = predicted - optimized
                cost_saved = saved * 8
                co2_saved = saved * 0.82
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Predicted Energy", f"{predicted:.1f} kWh")
                col2.metric("Optimized Energy", f"{optimized:.1f} kWh", delta=f"-{saved:.1f} kWh")
                col3.metric("Cost Savings", f"₹{cost_saved:.0f}/day")
                col4.metric("CO₂ Reduction", f"{co2_saved:.1f} kg/day")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted,
                    title={'text': "Predicted Energy (kWh)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, predicted*1.5]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, optimized], 'color': "lightgreen"},
                            {'range': [optimized, predicted], 'color': "orange"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': predicted
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
footer_cols = st.columns([3, 1, 1])
with footer_cols[0]:
    st.caption("📊 **Real Dataset Analysis Dashboard** | Data Source: Your Downloaded Datasets")
with footer_cols[1]:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d')}")
with footer_cols[2]:
    st.caption("🔬 Research Project")

# Add CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)