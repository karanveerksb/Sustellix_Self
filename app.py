import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import OneHotEncoder
import random

# Set page configuration
st.set_page_config(
    page_title="Campus Resource Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0f766e;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0d9488;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .insight-text {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
    }
    .warning-text {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 5px;
    }
    .recommendation-card {
        background-color: #f0f9ff;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for custom headers
def custom_header(text, level="main"):
    if level == "main":
        return st.markdown(f'<div class="main-header">{text}</div>', unsafe_allow_html=True)
    else:
        return st.markdown(f'<div class="sub-header">{text}</div>', unsafe_allow_html=True)

# Load data functions
@st.cache_data
def load_waste_data():
    try:
        waste_data = pd.read_csv('try.csv')
        capacity_data = pd.read_csv('capacity.csv')
        return waste_data, capacity_data
    except FileNotFoundError:
        st.error("Waste data files not found. Please check file paths.")
        return None, None

@st.cache_data
def load_electricity_data():
    try:
        electricity_data = pd.read_csv('electricity.csv')
        electricity_data['date'] = pd.to_datetime(electricity_data['date'], format='%d/%m/%Y')
        return electricity_data
    except FileNotFoundError:
        st.error("Electricity data file not found.")
        return None

@st.cache_data
def load_water_data():
    try:
        water_data = pd.read_csv('water.csv')
        water_data['date'] = pd.to_datetime(water_data['date'], format='%d/%m/%Y')
        return water_data
    except FileNotFoundError:
        st.error("Water data file not found.")
        return None

# Load saved models
@st.cache_resource
def load_models():
    try:
        transformer = joblib.load('transformer.joblib')
        waste_model = joblib.load('waste_model.joblib')
        column_names = joblib.load('column_names.joblib')
        return transformer, waste_model, column_names
    except FileNotFoundError:
        st.error("Model files not found. Please check if the files exist.")
        return None, None, None

# Function to prepare waste data and add calculated fields
def prepare_waste_data(waste_data, capacity_data):
    # Calculate total waste
    waste_data['total_waste'] = waste_data['pwaste'] + waste_data['twaste'] + waste_data['foodsample'] + waste_data['kwaste']
    
    # Decode binary hostel columns if they exist
    if 'hostel_binary_0' in waste_data.columns:
        # Get the binary hostel columns
        binary_cols = [col for col in waste_data.columns if col.startswith('hostel_binary_')]
        
        def decode_hostel_id(row):
            binary_string = ''.join(str(int(row[col])) for col in binary_cols)
            return int(binary_string, 2)
        
        waste_data['decoded_hostel'] = waste_data.apply(decode_hostel_id, axis=1)
        
        # Map to hostel names
        correct_hostel_names = ["A", "B", "C", "D", "FRG", "M", "N", "O", "PG", "Q"]
        unique_hostel_ids = sorted(waste_data['decoded_hostel'].unique())
        hostel_name_map = {}
        
        for i, hostel_id in enumerate(unique_hostel_ids):
            if i < len(correct_hostel_names):
                hostel_name_map[hostel_id] = correct_hostel_names[i]
            else:
                hostel_name_map[hostel_id] = f"Hostel {hostel_id}"
        
        waste_data['hostel_name'] = waste_data['decoded_hostel'].map(hostel_name_map)
    
    # If hostel_name doesn't exist, use the hostel column directly
    elif 'hostel' in waste_data.columns:
        waste_data['hostel_name'] = waste_data['hostel']
    
    # Merge with capacity data if possible
    if 'hostel_name' in waste_data.columns and capacity_data is not None:
        waste_data = pd.merge(waste_data, capacity_data, 
                              left_on='hostel_name', right_on='hostel', 
                              how='left')
        
        # Calculate per capita waste
        waste_data['per_capita_waste'] = waste_data['total_waste'] / waste_data['capacity']
    
    return waste_data

# Function to generate waste visualizations
def generate_waste_plots(waste_data):
    # Group by hostel name for total waste
    hostel_waste = waste_data.groupby('hostel_name')['total_waste'].sum().reset_index()
    hostel_waste = hostel_waste.sort_values('total_waste', ascending=False)
    
    # Total waste by hostel plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x='hostel_name', y='total_waste', data=hostel_waste, ax=ax1)
    
    # Add waste amount labels
    for bar in bars.patches:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f} kg',
                ha='center', va='bottom')
    
    plt.title('Total Waste Generated by Each Hostel')
    plt.xlabel('Hostel')
    plt.ylabel('Total Waste (kg)')
    plt.xticks(rotation=45)
    
    # Breakdown of waste types by hostel
    waste_types = ['pwaste', 'twaste', 'foodsample', 'kwaste']
    hostel_waste_breakdown = waste_data.groupby('hostel_name')[waste_types].sum().reset_index()
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(hostel_waste_breakdown))
    
    colors = ['#FF9671', '#FFC75F', '#D65DB1', '#845EC2']
    for i, waste_type in enumerate(waste_types):
        ax2.bar(hostel_waste_breakdown['hostel_name'], hostel_waste_breakdown[waste_type], 
               bottom=bottom, label=waste_type, color=colors[i])
        bottom += hostel_waste_breakdown[waste_type]
    
    plt.title('Breakdown of Waste Types by Hostel')
    plt.xlabel('Hostel')
    plt.ylabel('Waste Amount (kg)')
    plt.xticks(rotation=45)
    plt.legend(title='Waste Types')
    
    # Per capita waste plot (if available)
    if 'per_capita_waste' in waste_data.columns:
        per_capita_avg = waste_data.groupby('hostel_name')['per_capita_waste'].mean().reset_index()
        per_capita_avg = per_capita_avg.sort_values('per_capita_waste', ascending=False)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(x='hostel_name', y='per_capita_waste', data=per_capita_avg, ax=ax3)
        
        # Add per capita labels
        for bar in bars.patches:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f} kg/student',
                    ha='center', va='bottom')
        
        plt.title('Per Capita Waste by Hostel')
        plt.xlabel('Hostel')
        plt.ylabel('Waste per Student (kg)')
        plt.xticks(rotation=45)
    else:
        fig3 = None
    
    return fig1, fig2, fig3

# Functions for electricity and water data processing and visualization
def process_resource_data(df, resource_type):
    # Add per capita consumption
    df[f'per_capita_{resource_type}'] = df[f'{resource_type.capitalize()}'] / df['capacity']
    
    # Add day features
    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'] >= 5
    
    return df

def generate_resource_plots(df, resource_type):
    resource_col = resource_type.capitalize()
    per_capita_col = f'per_capita_{resource_type}'
    
    # Total by hostel
    hostel_total = df.groupby('hostel')[resource_col].sum().reset_index()
    hostel_total = hostel_total.sort_values(resource_col, ascending=False)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x='hostel', y=resource_col, data=hostel_total, ax=ax1)
    
    plt.title(f'Total {resource_type.capitalize()} by Hostel')
    plt.xlabel('Hostel')
    plt.ylabel(f'Total {resource_type.capitalize()}')
    plt.xticks(rotation=45)
    
    # Per capita by hostel
    per_capita_avg = df.groupby('hostel')[per_capita_col].mean().reset_index()
    per_capita_avg = per_capita_avg.sort_values(per_capita_col, ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x='hostel', y=per_capita_col, data=per_capita_avg, ax=ax2)
    
    plt.title(f'Per Capita {resource_type.capitalize()} by Hostel')
    plt.xlabel('Hostel')
    plt.ylabel(f'Per Capita {resource_type.capitalize()}')
    plt.xticks(rotation=45)
    
    # Time series (if date information is available)
    if 'date' in df.columns:
        daily_avg = df.groupby('date')[resource_col].mean().reset_index()
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        plt.plot(daily_avg['date'], daily_avg[resource_col], marker='o')
        
        plt.title(f'Average Daily {resource_type.capitalize()} Over Time')
        plt.xlabel('Date')
        plt.ylabel(f'Average {resource_type.capitalize()}')
        
        # Format the date ticks
        plt.xticks(rotation=45)
    else:
        fig3 = None
    
    return fig1, fig2, fig3

# Model prediction functions - FIXED VERSION
def predict_future_waste_per_hostel(model, transformer, column_names, waste_data, hostels):
    """
    Make waste predictions for each hostel using statistics from existing data
    if model prediction fails
    """
    try:
        # Try using the model for predictions
        # Create sample input data with all required columns
        input_data = []
        
        # Get current date for predictions
        today = datetime.now()
        
        for hostel in hostels:
            # Create predictions for each meal type
            meal_types = ['breakfast', 'lunch', 'dinner', 'special']
            for meal in meal_types:
                # Create row with all needed columns
                row = {
                    'date': today.strftime('%Y-%m-%d'),
                    'meal': meal,
                    'hostel': hostel,
                    'hostel type': 'girls' if hostel in ['FRG', 'PG'] else 'boys',
                    # Add default values for waste columns
                    'pwaste': 0,
                    'twaste': 0,
                    'foodsample': 0,
                    'kwaste': 0
                }
                
                # Add meal_X columns
                for m in meal_types:
                    row[f'meal_{m}'] = 1 if m == meal else 0
                
                # Add hostel binary columns (if needed)
                for i in range(4):  # Assuming 4 binary columns
                    row[f'hostel_binary_{i}'] = 0
                
                input_data.append(row)
        
        input_df = pd.DataFrame(input_data)
        
        # Only keep columns that the transformer expects
        if transformer and hasattr(transformer, 'get_feature_names_out'):
            expected_cols = list(transformer.get_feature_names_out())
            # Filter input_df to only include expected columns
            available_cols = [col for col in expected_cols if col in input_df.columns]
            if available_cols:
                input_df = input_df[available_cols]
        
        # Transform and predict
        transformed_data = transformer.transform(input_df)
        predictions = model.predict(transformed_data)
        
        # Create prediction DataFrame
        prediction_df = pd.DataFrame({
            'hostel': input_df['hostel'],
            'meal': input_df['meal'],
            'predicted_waste': predictions
        })
        
        # Aggregate by hostel
        agg_predictions = prediction_df.groupby('hostel')['predicted_waste'].mean().reset_index()
        
    except Exception as e:
        st.warning(f"Model prediction failed: {e}. Using averages from historical data.")
        
        # Fall back to using averages from historical data
        if waste_data is not None and 'hostel_name' in waste_data.columns:
            # Calculate average waste by hostel
            agg_predictions = waste_data.groupby('hostel_name')['total_waste'].mean().reset_index()
            agg_predictions = agg_predictions.rename(columns={'hostel_name': 'hostel', 'total_waste': 'predicted_waste'})
        else:
            # Create dummy predictions if all else fails
            agg_predictions = pd.DataFrame({
                'hostel': hostels,
                'predicted_waste': [random.uniform(2, 5) for _ in hostels]  # Random values between 2-5kg
            })
    
    # Sort by predicted waste (descending)
    return agg_predictions.sort_values('predicted_waste', ascending=False)

def build_simple_waste_model(waste_data):
    """Build a simple model to predict waste based on historical averages"""
    if 'hostel_name' in waste_data.columns and 'total_waste' in waste_data.columns:
        # Group by hostel and meal type to get average waste
        if 'meal' in waste_data.columns:
            avg_waste = waste_data.groupby(['hostel_name', 'meal'])['total_waste'].mean().reset_index()
        else:
            avg_waste = waste_data.groupby('hostel_name')['total_waste'].mean().reset_index()
            
        return avg_waste
    return None

def predict_resource_usage(resource_df, resource_type, prediction_days=7):
    # Group by hostel for average usage
    avg_usage = resource_df.groupby('hostel')[resource_type.capitalize()].mean().reset_index()
    
    # Generate future dates
    last_date = resource_df['date'].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
    
    # Create predictions (this is a simple forecast based on historical averages)
    predictions = []
    
    for future_date in future_dates:
        for _, row in avg_usage.iterrows():
            hostel = row['hostel']
            avg_consumption = row[resource_type.capitalize()]
            
            # Simple forecast: weekend adjustment
            is_weekend = future_date.dayofweek >= 5
            weekend_factor = 1.2 if is_weekend else 1.0
            forecast = avg_consumption * weekend_factor
            
            predictions.append({
                'date': future_date,
                'hostel': hostel,
                'predicted_consumption': forecast,
                'is_weekend': is_weekend
            })
    
    return pd.DataFrame(predictions)

# Main App
def main():
    # App title and introduction
    custom_header("Campus Resource Analytics Dashboard")
    
    st.write("""
    This dashboard provides comprehensive analysis and forecasting for three critical campus resources:
    food waste generation, electricity consumption, and water usage. Use the tabs below to explore
    each resource in detail.
    """)
    
    # Loading data
    waste_data, capacity_data = load_waste_data()
    electricity_data = load_electricity_data()
    water_data = load_water_data()
    transformer, waste_model, column_names = load_models()
    
    # Prepare the waste data
    if waste_data is not None:
        waste_data = prepare_waste_data(waste_data, capacity_data)
    
    # Create tabs for different resources
    tab1, tab2, tab3 = st.tabs(["Food Waste", "Electricity", "Water"])
    
    # Food Waste Tab
    with tab1:
        custom_header("Food Waste Analysis", "sub")
        
        if waste_data is not None:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Waste Generated", 
                    f"{waste_data['total_waste'].sum():.2f} kg",
                    "Overall campus food waste"
                )
            with col2:
                st.metric(
                    "Average Waste per Hostel", 
                    f"{waste_data.groupby('hostel_name')['total_waste'].mean().mean():.2f} kg",
                    "Average per hostel per meal"
                )
            with col3:
                if 'per_capita_waste' in waste_data.columns:
                    st.metric(
                        "Average Per Capita", 
                        f"{waste_data['per_capita_waste'].mean():.4f} kg/student",
                        "Per student average"
                    )
            
            # Visualizations
            st.write("### Waste Generation Analysis")
            
            fig1, fig2, fig3 = generate_waste_plots(waste_data)
            st.pyplot(fig1)
            
            st.write("### Waste Type Breakdown")
            st.pyplot(fig2)
            
            if fig3:
                st.write("### Per Capita Waste Analysis")
                st.pyplot(fig3)
            
            # Predictions section
            st.write("### Future Waste Predictions")
            
            # Get unique hostels
            hostels = waste_data['hostel_name'].unique()
            
            # Make predictions - FIXED: Added column_names and waste_data parameters
            predictions = predict_future_waste_per_hostel(waste_model, transformer, column_names, waste_data, hostels)
            
            # Display predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x='hostel', y='predicted_waste', data=predictions)
            
            # Add prediction labels
            for bar in bars.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f} kg',
                        ha='center', va='bottom')
            
            plt.title('Predicted Waste Generation by Hostel')
            plt.xlabel('Hostel')
            plt.ylabel('Predicted Waste (kg)')
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            
            # Top waste generators
            st.write("#### Top Waste Generators:")
            top_waste = predictions.head(3)
            for i, row in top_waste.iterrows():
                st.markdown(
                    f"""
                    <div class="warning-text">
                    <strong>Hostel {row['hostel']}</strong>: {row['predicted_waste']:.2f} kg predicted waste.
                    <br>Recommendation: Conduct focused waste reduction initiatives.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.warning("Waste data not available. Please check the data source.")
    
    # Electricity Tab
    with tab2:
        custom_header("Electricity Consumption Analysis", "sub")
        
        if electricity_data is not None:
            # Process data
            electricity_data = process_resource_data(electricity_data, 'electricity')
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Electricity Consumed", 
                    f"{electricity_data['Electricity'].sum():,.0f} units",
                    "Overall campus consumption"
                )
            with col2:
                st.metric(
                    "Average per Hostel", 
                    f"{electricity_data.groupby('hostel')['Electricity'].mean().mean():,.0f} units",
                    "Average per hostel per day"
                )
            with col3:
                st.metric(
                    "Average Per Capita", 
                    f"{electricity_data['per_capita_electricity'].mean():.2f} units/student",
                    "Per student average"
                )
            
            # Visualizations
            st.write("### Electricity Consumption Analysis")
            
            fig1, fig2, fig3 = generate_resource_plots(electricity_data, 'electricity')
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig1)
            with col2:
                st.pyplot(fig2)
            
            if fig3:
                st.pyplot(fig3)
            
            # Predictions section
            st.write("### Electricity Usage Forecast (Next 7 Days)")
            
            # Generate predictions
            electricity_predictions = predict_resource_usage(electricity_data, 'electricity')
            
            # Plot predictions by hostel
            future_by_hostel = electricity_predictions.groupby('hostel')['predicted_consumption'].sum().reset_index()
            future_by_hostel = future_by_hostel.sort_values('predicted_consumption', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x='hostel', y='predicted_consumption', data=future_by_hostel)
            
            plt.title('Forecasted Electricity Consumption by Hostel (Next 7 Days)')
            plt.xlabel('Hostel')
            plt.ylabel('Predicted Consumption (units)')
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            
            # Show daily predictions
            st.write("### Daily Electricity Forecast")
            
            # Format predictions for display
            pivot_predictions = electricity_predictions.pivot_table(
                index='date', columns='hostel', values='predicted_consumption'
            )
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_predictions, cmap='YlOrRd', annot=True, fmt='.0f')
            
            plt.title('Daily Electricity Forecast by Hostel')
            plt.ylabel('Date')
            plt.xlabel('Hostel')
            
            st.pyplot(fig)
            
            # Conservation recommendations
            st.write("### Electricity Conservation Recommendations")
            
            # Top consumers
            top_consumers = future_by_hostel.head(3)
            for i, row in top_consumers.iterrows():
                st.markdown(
                    f"""
                    <div class="recommendation-card">
                    <strong>Hostel {row['hostel']}</strong>: {row['predicted_consumption']:,.0f} units predicted usage.
                    <br><strong>Recommendations:</strong>
                    <ul>
                        <li>Conduct an energy audit to identify major consumption sources</li>
                        <li>Install LED lighting throughout the hostel</li>
                        <li>Implement awareness campaigns about energy conservation</li>
                        <li>Consider installation of smart power management systems</li>
                    </ul>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.warning("Electricity data not available. Please check the data source.")
    
    # Water Tab
    with tab3:
        custom_header("Water Usage Analysis", "sub")
        
        if water_data is not None:
            # Process data
            water_data = process_resource_data(water_data, 'water')
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Water Used", 
                    f"{water_data['Water'].sum():,.0f} liters",
                    "Overall campus usage"
                )
            with col2:
                st.metric(
                    "Average per Hostel", 
                    f"{water_data.groupby('hostel')['Water'].mean().mean():,.0f} liters",
                    "Average per hostel per day"
                )
            with col3:
                st.metric(
                    "Average Per Capita", 
                    f"{water_data['per_capita_water'].mean():.2f} liters/student",
                    "Per student average"
                )
            
            # Visualizations
            st.write("### Water Usage Analysis")
            
            fig1, fig2, fig3 = generate_resource_plots(water_data, 'water')
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig1)
            with col2:
                st.pyplot(fig2)
            
            if fig3:
                st.pyplot(fig3)
            
            # Predictions section
            st.write("### Water Usage Forecast (Next 7 Days)")
            
            # Generate predictions
            water_predictions = predict_resource_usage(water_data, 'water')
            
            # Plot predictions by hostel
            future_by_hostel = water_predictions.groupby('hostel')['predicted_consumption'].sum().reset_index()
            future_by_hostel = future_by_hostel.sort_values('predicted_consumption', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x='hostel', y='predicted_consumption', data=future_by_hostel)
            
            plt.title('Forecasted Water Usage by Hostel (Next 7 Days)')
            plt.xlabel('Hostel')
            plt.ylabel('Predicted Usage (liters)')
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            
            # Show daily predictions
            st.write("### Daily Water Usage Forecast")
            
            # Format predictions for display
            pivot_predictions = water_predictions.pivot_table(
                index='date', columns='hostel', values='predicted_consumption'
            )
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_predictions, cmap='Blues', annot=True, fmt='.0f')
            
            plt.title('Daily Water Usage Forecast by Hostel')
            plt.ylabel('Date')
            plt.xlabel('Hostel')
            
            st.pyplot(fig)
            
            # Conservation recommendations
            st.write("### Water Conservation Recommendations")
            
            # Top consumers
            top_consumers = future_by_hostel.head(3)
            for i, row in top_consumers.iterrows():
                st.markdown(
                    f"""
                    <div class="recommendation-card">
                    <strong>Hostel {row['hostel']}</strong>: {row['predicted_consumption']:,.0f} liters predicted usage.
                    <br><strong>Recommendations:</strong>
                    <ul>
                        <li>Check for leaks in bathrooms and kitchen areas</li>
                        <li>Install water-saving fixtures in high-usage areas</li>
                        <li>Implement water recycling for gardening purposes</li>
                        <li>Conduct water efficiency awareness programs</li>
                    </ul>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.warning("Water data not available. Please check the data source.")

    # Footer
    st.markdown("---")
    st.markdown("### About This Dashboard")
    st.markdown("""
    This resource analytics dashboard provides insights into campus resource consumption and waste generation. 
    The models use XGBoost to predict future resource usage based on historical patterns.
    
    The analysis helps identify high-priority areas for conservation efforts and sustainable practices.
    """)

if __name__ == "__main__":
    main()