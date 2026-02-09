# dashboard/app.py - SHOWS REAL DATASETS
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json

# Page config
st.set_page_config(
    page_title="Smart Campus Energy Optimizer",
    page_icon="⚡",
    layout="wide"
)

# Title
st.title("⚡ Smart Campus Energy Optimizer")
st.markdown("### Real Dataset Analysis Dashboard")

# ==================== LOAD REAL DATASETS ====================
@st.cache_data
def load_real_datasets():
    """Load all available real datasets"""
    datasets = {}
    
    # Paths to your real datasets
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
with st.sidebar:
    st.header("📊 Data Selection")
    
    # Load datasets
    with st.spinner("Loading real datasets..."):
        datasets = load_real_datasets()
    
    if datasets:
        selected_dataset = st.selectbox(
            "Choose Dataset",
            list(datasets.keys())
        )
        
        df = datasets[selected_dataset]
        
        # Show dataset info
        st.info(f"""
        **Dataset:** {selected_dataset}
        **Records:** {len(df):,}
        **Columns:** {len(df.columns)}
        **Size:** {df.memory_usage(deep=True).sum()/(1024*1024):.1f} MB
        """)
        
        # Time range selector (if timestamp exists)
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
    st.header("🎯 Analysis Type")
    analysis_type = st.radio(
        "Select Analysis",
        ["📈 Energy Patterns", "🏢 Building Comparison", "💰 Savings Analysis", "🌡️ Environmental Impact"]
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
    # Show dataset preview
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
    
    # ==================== ANALYSIS 1: ENERGY PATTERNS ====================
    if analysis_type == "📈 Energy Patterns":
        st.header("📈 Energy Consumption Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Find energy column
            energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'kwh' in col.lower() or 'meter' in col.lower()]
            
            if energy_cols:
                energy_col = st.selectbox("Select Energy Column", energy_cols)
                
                # Daily pattern
                if 'hour_of_day' in df.columns or 'hour' in df.columns:
                    hour_col = 'hour_of_day' if 'hour_of_day' in df.columns else 'hour'
                    daily_pattern = df.groupby(hour_col)[energy_col].mean().reset_index()
                    
                    fig1 = px.line(daily_pattern, x=hour_col, y=energy_col, 
                                  title="📅 Average Daily Energy Pattern",
                                  markers=True)
                    fig1.update_layout(xaxis_title="Hour of Day", yaxis_title="Energy (kWh)")
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Weekly pattern
                if 'day_of_week' in df.columns:
                    weekly_pattern = df.groupby('day_of_week')[energy_col].mean().reset_index()
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    
                    fig2 = px.bar(weekly_pattern, x='day_of_week', y=energy_col,
                                 title="📆 Weekly Energy Pattern",
                                 labels={'day_of_week': 'Day', energy_col: 'Energy (kWh)'})
                    fig2.update_xaxes(ticktext=days, tickvals=list(range(7)))
                    st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Histogram of energy consumption
            if energy_cols:
                fig3 = px.histogram(df, x=energy_cols[0], 
                                   title="📊 Energy Distribution",
                                   nbins=50)
                fig3.update_layout(xaxis_title="Energy (kWh)", yaxis_title="Frequency")
                st.plotly_chart(fig3, use_container_width=True)
                
                # Statistics
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
    
    # ==================== ANALYSIS 2: BUILDING COMPARISON ====================
    elif analysis_type == "🏢 Building Comparison":
        st.header("🏢 Building Energy Comparison")
        
        # Find building column
        building_cols = [col for col in df.columns if 'building' in col.lower() or 'site' in col.lower()]
        energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'kwh' in col.lower()]
        
        if building_cols and energy_cols:
            building_col = st.selectbox("Select Building Column", building_cols)
            energy_col = st.selectbox("Select Energy Column", energy_cols)
            
            # Building comparison
            building_stats = df.groupby(building_col)[energy_col].agg(['mean', 'std', 'count']).round(2)
            building_stats = building_stats.sort_values('mean', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig1 = px.bar(building_stats.reset_index(), 
                            x=building_col, y='mean',
                            title="🏢 Average Energy by Building",
                            error_y='std')
                fig1.update_layout(xaxis_title="Building", yaxis_title="Average Energy (kWh)")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Display statistics
                st.subheader("📊 Building Statistics")
                st.dataframe(building_stats.head(10), use_container_width=True)
                
                # Calculate optimization potential
                st.subheader("🎯 Optimization Potential")
                avg_energy = df[energy_col].mean()
                potential_savings = avg_energy * 0.15  # 15% savings
                daily_cost_savings = potential_savings * 8  # ₹8 per kWh
                monthly_savings = daily_cost_savings * 30
                
                st.metric("Average Energy", f"{avg_energy:.1f} kWh")
                st.metric("Potential Savings (15%)", f"{potential_savings:.1f} kWh/day")
                st.metric("Cost Savings", f"₹{daily_cost_savings:.0f}/day")
                st.metric("Monthly Impact", f"₹{monthly_savings:,.0f}")
    
    # ==================== ANALYSIS 3: SAVINGS ANALYSIS ====================
    elif analysis_type == "💰 Savings Analysis":
        st.header("💰 Cost & Savings Analysis")
        
        # Calculate savings based on real data
        energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'kwh' in col.lower()]
        
        if energy_cols:
            energy_col = energy_cols[0]
            avg_energy = df[energy_col].mean()
            
            # Savings calculation
            savings_percent = st.slider("Optimization Percentage", 5, 25, 15) / 100
            
            current_daily = avg_energy
            optimized_daily = current_daily * (1 - savings_percent)
            energy_saved = current_daily - optimized_daily
            
            # Financial calculations
            electricity_rate = st.number_input("Electricity Rate (₹/kWh)", 5.0, 15.0, 8.0, 0.5)
            daily_cost_saved = energy_saved * electricity_rate
            monthly_cost_saved = daily_cost_saved * 30
            annual_cost_saved = daily_cost_saved * 365
            
            # CO₂ calculations
            co2_per_kwh = st.number_input("CO₂ per kWh (kg)", 0.5, 1.5, 0.82, 0.01)
            daily_co2_saved = energy_saved * co2_per_kwh
            annual_co2_saved = daily_co2_saved * 365
            trees_equivalent = annual_co2_saved / 20  # 1 tree absorbs 20 kg/year
            
            # Display metrics
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
            
            # Detailed breakdown
            st.subheader("📊 Detailed Impact Analysis")
            
            breakdown_data = {
                'Period': ['Daily', 'Monthly', 'Annual'],
                'Energy Saved (kWh)': [energy_saved, energy_saved*30, energy_saved*365],
                'Cost Savings (₹)': [daily_cost_saved, monthly_cost_saved, annual_cost_saved],
                'CO₂ Reduced (kg)': [daily_co2_saved, daily_co2_saved*30, annual_co2_saved]
            }
            
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True)
            
            # Visualization
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
            
            fig.update_layout(
                title='Energy Optimization Impact',
                yaxis_title='Energy (kWh/day)',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== ANALYSIS 4: ENVIRONMENTAL IMPACT ====================
    else:  # Environmental Impact
        st.header("🌱 Environmental Impact Analysis")
        
        energy_cols = [col for col in df.columns if 'energy' in col.lower() or 'kwh' in col.lower()]
        
        if energy_cols:
            energy_col = energy_cols[0]
            total_energy = df[energy_col].sum()
            
            # Environmental calculations
            co2_per_kwh = 0.82  # kg CO₂ per kWh (India average)
            total_co2 = total_energy * co2_per_kwh
            
            # Equivalent metrics
            trees_needed = total_co2 / 20  # 1 tree absorbs 20 kg/year
            cars_equivalent = total_co2 / 4200  # 1 car emits 4200 kg/year
            flights_equivalent = total_co2 / 200  # 1 Delhi-Mumbai flight emits 200 kg
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Energy", f"{total_energy:,.0f} kWh")
            with col2:
                st.metric("CO₂ Emissions", f"{total_co2:,.0f} kg")
            with col3:
                st.metric("Trees Needed", f"{trees_needed:.0f} trees")
            with col4:
                st.metric("Car Equivalents", f"{cars_equivalent:.1f} cars/year")
            
            # Environmental visualization
            st.subheader("🌍 Carbon Footprint Analysis")
            
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
            
            # Show research citations
            citations_path = "data/raw/research_citations.csv"
            if os.path.exists(citations_path):
                st.subheader("📚 Research Basis")
                citations = pd.read_csv(citations_path)
                for _, row in citations.iterrows():
                    with st.expander(f"📄 {row['study']} ({row['year']})"):
                        st.write(f"**Finding:** {row['key_finding']}")
                        st.write(f"**Source:** {row['source']}")

# ==================== FOOTER ====================
st.markdown("---")
footer_cols = st.columns([3, 1, 1])
with footer_cols[0]:
    st.caption("📊 **Real Dataset Analysis Dashboard** | Data Source: Your Downloaded Datasets")
with footer_cols[1]:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d')}")
with footer_cols[2]:
    st.caption("🔬 Research Project")

# Add CSS for better appearance
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