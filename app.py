# Streamlit App for Customer Purchasing Behavior Analysis
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Set page configuration
st.set_page_config(
    page_title="Customer Purchasing Behavior Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance - Changed background to dark and adjusted text colors for visibility
st.markdown("""
<style>
    .main {
        background-color: #121212; /* Dark background */
        color: #f0f0f0; /* Light text for better contrast */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #7fdbff; /* Light blue headers for visibility on dark background */
    }
    .stAlert {
        background-color: #1e2a38; /* Darker alert background */
        border-color: #7fdbff;
        color: #f0f0f0;
    }
    div[data-testid="stSidebar"] {
        background-color: #1e2a38; /* Dark sidebar */
        color: #f0f0f0;
    }
    /* Ensure text inputs and selectboxes have good contrast */
    .stTextInput, .stSelectbox {
        background-color: #333333;
        color: white !important;
    }
    /* Fix for dataframe/table readability */
    .dataframe {
        background-color: #2d3748 !important;
        color: #f0f0f0 !important;
    }
    .dataframe th {
        background-color: #2d3748 !important;
        color: #7fdbff !important;
    }
    /* Make metric values more visible */
    [data-testid="stMetricValue"] {
        color: #7fdbff !important;
    }
    /* Info boxes */
    .stInfo {
        background-color: #1e2a38 !important;
        color: #f0f0f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_data():
    if os.path.exists('Customer_Purchasing_Behaviors.csv'):
        df = pd.read_csv('Customer_Purchasing_Behaviors.csv')
        return df
    else:
        st.error("Dataset file not found. Please upload the Customer_Purchasing_Behaviors.csv file.")
        return None

@st.cache_resource
def load_models():
    models = {}
    try:
        models['kmeans'] = joblib.load('kmeans_model.pkl')
        models['rf_model'] = joblib.load('rf_model.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
        return models
    except:
        st.warning("Models not found. Please run the analysis first.")
        return None

# Main function to run the app
def main():
    # Sidebar
    st.sidebar.title("Customer Analysis Dashboard")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select a page:",
        ["Home", "Data Exploration", "Customer Segmentation", "Loyalty Prediction", "Business Insights"]
    )
    
    # Load data
    df = load_data()
    
    if df is None:
        st.file_uploader("Upload Customer Data CSV", type=["csv"])
        return
    
    # Load models if available
    models = load_models()
    
    # Page content
    if page == "Home":
        show_home_page(df)
    elif page == "Data Exploration":
        show_data_exploration(df)
    elif page == "Customer Segmentation":
        show_customer_segmentation(df, models)
    elif page == "Loyalty Prediction":
        show_loyalty_prediction(df, models)
    elif page == "Business Insights":
        show_business_insights(df, models)

def show_home_page(df):
    st.title("Customer Purchasing Behavior Analysis")
    
    st.markdown("""
    ## Welcome to the Customer Analysis Dashboard!
    
    This interactive dashboard helps you analyze customer purchasing behaviors, identify customer segments,
    predict loyalty scores, and gain actionable business insights.
    
    ### Dataset Overview
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", f"{len(df)}")
    with col2:
        st.metric("Average Purchase", f"${df['purchase_amount'].mean():.2f}")
    with col3:
        st.metric("Average Loyalty Score", f"{df['loyalty_score'].mean():.2f}")
    
    st.markdown("### Sample Data")
    st.dataframe(df.head())
    
    st.markdown("""
    ### Available Features:
    
    - **Data Exploration**: Analyze distributions and relationships among variables
    - **Customer Segmentation**: Identify distinct customer segments
    - **Loyalty Prediction**: Predict customer loyalty scores
    - **Business Insights**: Get actionable insights for marketing and customer retention
    
    Use the sidebar to navigate between pages.
    """)
    
    # Key metrics visualization
    st.subheader("Key Metrics by Region")
    region_metrics = df.groupby('region').agg({
        'purchase_amount': 'mean',
        'loyalty_score': 'mean',
        'purchase_frequency': 'mean'
    }).reset_index()
    
    fig = px.bar(region_metrics, x='region', y=['purchase_amount', 'loyalty_score', 'purchase_frequency'],
                barmode='group', title='Average Metrics by Region',
                labels={'value': 'Value', 'variable': 'Metric'},
                template="plotly_dark")  # Use dark template for better visibility
    st.plotly_chart(fig, use_container_width=True)

def show_data_exploration(df):
    st.title("Data Exploration")
    
    st.markdown("### Explore the distributions and relationships in the data")
    
    # Data summary
    if st.checkbox("Show Data Summary"):
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
    
    # Distribution plots
    st.subheader("Variable Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(df, x='age', marginal='box', title='Age Distribution', 
                             color_discrete_sequence=px.colors.qualitative.Vivid,
                             template="plotly_dark")  # Dark theme for better visibility
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Purchase amount distribution
        fig_purchase = px.histogram(df, x='purchase_amount', marginal='box', 
                                  title='Purchase Amount Distribution',
                                  color_discrete_sequence=px.colors.qualitative.Vivid,
                                  template="plotly_dark")  # Dark theme
        st.plotly_chart(fig_purchase, use_container_width=True)
        
    with col2:
        # Annual income distribution
        fig_income = px.histogram(df, x='annual_income', marginal='box', 
                                title='Annual Income Distribution',
                                color_discrete_sequence=px.colors.qualitative.Vivid,
                                template="plotly_dark")  # Dark theme
        st.plotly_chart(fig_income, use_container_width=True)
        
        # Loyalty score distribution
        fig_loyalty = px.histogram(df, x='loyalty_score', marginal='box', 
                                 title='Loyalty Score Distribution',
                                 color_discrete_sequence=px.colors.qualitative.Vivid,
                                 template="plotly_dark")  # Dark theme
        st.plotly_chart(fig_loyalty, use_container_width=True)
    
    # Regional analysis
    st.subheader("Regional Analysis")
    
    # Region distribution
    region_counts = df['region'].value_counts().reset_index()
    region_counts.columns = ['region', 'count']
    
    fig_region = px.pie(region_counts, values='count', names='region', 
                      title='Customer Distribution by Region',
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      template="plotly_dark")  # Dark theme
    st.plotly_chart(fig_region, use_container_width=True)
    
    # Metrics by region
    st.subheader("Purchase Metrics by Region")
    
    metric_option = st.selectbox(
        "Select metric to visualize:",
        ["purchase_amount", "loyalty_score", "purchase_frequency"]
    )
    
    fig_box = px.box(df, x='region', y=metric_option, 
                   title=f'{metric_option.replace("_", " ").title()} by Region',
                   color='region',
                   color_discrete_sequence=px.colors.qualitative.Bold,
                   template="plotly_dark")  # Dark theme
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Correlation Analysis")
    
    corr = df.select_dtypes(include=[np.number]).corr()
    fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto",
                       title="Correlation Matrix",
                       color_continuous_scale='Viridis',
                       template="plotly_dark")  # Dark theme
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Scatter plots for relationships
    st.subheader("Relationship Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X variable:", df.select_dtypes(include=[np.number]).columns)
    
    with col2:
        y_var = st.selectbox("Select Y variable:", 
                           [col for col in df.select_dtypes(include=[np.number]).columns if col != x_var],
                           index=1 if len(df.select_dtypes(include=[np.number]).columns) > 1 else 0)
    
    fig_scatter = px.scatter(df, x=x_var, y=y_var, color='region',
                          title=f'{x_var.replace("_", " ").title()} vs {y_var.replace("_", " ").title()}',
                          trendline="ols", opacity=0.7,
                          color_discrete_sequence=px.colors.qualitative.Bold,
                          template="plotly_dark")  # Dark theme
    st.plotly_chart(fig_scatter, use_container_width=True)

def show_customer_segmentation(df, models):
    st.title("Customer Segmentation")
    
    if models is None:
        st.warning("Models not loaded. Please run the analysis first.")
        return
    
    st.markdown("""
    ### Customer Segmentation Analysis
    
    Identifying distinct customer segments helps in developing targeted marketing strategies.
    The K-means clustering algorithm has been used to create these segments.
    """)
    
    # Preprocessing for clustering
    df_encoded = pd.get_dummies(df, columns=['region'], drop_first=True)
    X = df_encoded.drop(['user_id'], axis=1)
    
    # Apply scaling
    scaler = models['scaler']
    X_scaled = scaler.transform(X)
    
    # Apply clustering
    kmeans = models['kmeans']
    cluster_labels = kmeans.predict(X_scaled)
    
    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Display cluster information
    st.subheader("Cluster Profiles")
    
    cluster_info = df_with_clusters.groupby('cluster').agg({
        'age': 'mean',
        'annual_income': 'mean',
        'purchase_amount': 'mean',
        'loyalty_score': 'mean',
        'purchase_frequency': 'mean',
        'user_id': 'count'
    }).rename(columns={'user_id': 'count'})
    
    st.dataframe(cluster_info.style.highlight_max(axis=0, color='lightgreen').format({
        'age': '{:.1f}',
        'annual_income': '${:.2f}',
        'purchase_amount': '${:.2f}',
        'loyalty_score': '{:.2f}',
        'purchase_frequency': '{:.1f}',
        'count': '{:.0f}'
    }))
    
    # Visualization of clusters
    st.subheader("Cluster Visualization")
    
    # Create a 3D scatter plot of clusters
    features_for_viz = st.multiselect(
        "Select features for visualization (choose 3):",
        options=df.select_dtypes(include=[np.number]).columns.tolist(),
        default=['age', 'annual_income', 'purchase_amount'][:3]
    )
    
    if len(features_for_viz) == 3:
        fig = px.scatter_3d(
            df_with_clusters, 
            x=features_for_viz[0], 
            y=features_for_viz[1],
            z=features_for_viz[2],
            color='cluster',
            title=f'Customer Clusters based on {", ".join(features_for_viz)}',
            labels={
                features_for_viz[0]: features_for_viz[0].replace('_', ' ').title(),
                features_for_viz[1]: features_for_viz[1].replace('_', ' ').title(),
                features_for_viz[2]: features_for_viz[2].replace('_', ' ').title()
            },
            opacity=0.7,
            template="plotly_dark"  # Dark theme
        )
        fig.update_layout(legend_title_text='Cluster')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select exactly 3 features for the 3D visualization.")
    
    # Radar chart for cluster comparison
    st.subheader("Cluster Comparison")
    
    # Normalize the data for radar chart
    radar_df = cluster_info.copy()
    radar_df = radar_df.drop(columns=['count'])
    
    # Scale the data between 0 and 1 for each feature
    for col in radar_df.columns:
        radar_df[col] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min())
    
    # Create radar chart
    categories = radar_df.columns.tolist()
    fig_radar = go.Figure()
    
    for cluster in radar_df.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df.loc[cluster].values.tolist(),
            theta=categories,
            fill='toself',
            name=f'Cluster {cluster}'
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Normalized Cluster Characteristics",
        template="plotly_dark",  # Dark theme
        paper_bgcolor="#121212",  # Dark background
        plot_bgcolor="#121212"    # Dark background
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Region distribution in clusters
    st.subheader("Region Distribution in Clusters")
    
    region_cluster = pd.crosstab(df_with_clusters['region'], df_with_clusters['cluster'])
    region_cluster_pct = region_cluster.div(region_cluster.sum(axis=0), axis=1) * 100
    
    fig_region_cluster = px.bar(
        region_cluster_pct.reset_index().melt(id_vars='region', var_name='cluster', value_name='percentage'),
        x='cluster',
        y='percentage',
        color='region',
        title="Region Distribution within Clusters (%)",
        labels={'percentage': 'Percentage (%)'},
        barmode='stack',
        template="plotly_dark"  # Dark theme
    )
    
    st.plotly_chart(fig_region_cluster, use_container_width=True)
    
    # Download cluster data
    if st.button("Download Segmented Customer Data"):
        csv = df_with_clusters.to_csv(index=False)
        st.download_button(
            label="Click to Download",
            data=csv,
            file_name="segmented_customers.csv",
            mime="text/csv"
        )

def show_loyalty_prediction(df, models):
    st.title("Loyalty Score Prediction")
    
    if models is None:
        st.warning("Models not loaded. Please run the analysis first.")
        return
    
    st.markdown("""
    ### Predict Customer Loyalty
    
    This tool allows you to predict a customer's loyalty score based on their demographic and purchasing information.
    """)
    
    # Create input form for prediction
    st.subheader("Enter Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        annual_income = st.number_input("Annual Income ($)", min_value=20000, max_value=100000, value=50000)
    
    with col2:
        purchase_amount = st.number_input("Purchase Amount ($)", min_value=100, max_value=1000, value=300)
        purchase_frequency = st.number_input("Purchase Frequency (times/year)", min_value=5, max_value=30, value=15)
    
    with col3:
        region = st.selectbox("Region", options=["North", "South", "East", "West"])
    
    # Create feature vector for prediction
    if st.button("Predict Loyalty Score"):
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'age': [age],
            'annual_income': [annual_income],
            'purchase_amount': [purchase_amount],
            'purchase_frequency': [purchase_frequency],
            'region': [region]
        })
        
        # Create dummy sample data to get the correct structure
        # We don't need to drop 'loyalty_score' here because the scaler was trained with it included
        sample_data = df.drop(['user_id'], axis=1, errors='ignore')
        
        # One-hot encode the region for both sample and input data
        sample_encoded = pd.get_dummies(sample_data, columns=['region'], drop_first=True)
        input_encoded = pd.get_dummies(input_data, columns=['region'], drop_first=True)
        
        # Make sure input_encoded has the same columns as sample_encoded (except loyalty_score)
        for col in sample_encoded.columns:
            if col != 'loyalty_score' and col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Keep only the columns that were in the training data and ensure same order
        # But exclude the target variable 'loyalty_score'
        input_cols = [col for col in sample_encoded.columns if col != 'loyalty_score']
        input_encoded = input_encoded[input_cols]
        
        # Scale the features
        scaler = models['scaler']
        X_scaled = scaler.transform(input_encoded)
        
        # Make prediction
        rf_model = models['rf_model']
        prediction = rf_model.predict(X_scaled)[0]
        
        # Display the result
        st.success(f"Predicted Loyalty Score: {prediction:.2f}")
        
        # Interpret the result
        if prediction >= 8.0:
            loyalty_category = "High Loyalty"
            description = "This customer is highly loyal and likely to continue purchasing. Consider offering premium services or loyalty rewards."
        elif prediction >= 5.0:
            loyalty_category = "Medium Loyalty"
            description = "This customer shows moderate loyalty. Focus on enhancing their experience and offering targeted promotions."
        else:
            loyalty_category = "Low Loyalty"
            description = "This customer is at risk of churn. Consider re-engagement campaigns and special offers."
        
        st.info(f"**Loyalty Category**: {loyalty_category}\n\n{description}")
        
        # Show feature importance
        st.subheader("What Factors Impact Loyalty?")
        
        # Use the correct feature names - exclude 'loyalty_score'
        feature_names = [col for col in sample_encoded.columns if col != 'loyalty_score']
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance for Loyalty Prediction',
            color='Importance',
            color_continuous_scale='Viridis',
            template="plotly_dark"  # Dark theme
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Show loyalty distribution
        st.subheader("Loyalty Score Distribution")
        
        fig_loyalty_dist = px.histogram(
            df,
            x='loyalty_score',
            nbins=20,
            title="Distribution of Loyalty Scores",
            color_discrete_sequence=['#3366CC'],
            template="plotly_dark"  # Dark theme
        )
        
        fig_loyalty_dist.add_vline(x=prediction, line_dash="dash", line_color="red", annotation_text="Prediction")
        
        st.plotly_chart(fig_loyalty_dist, use_container_width=True)

def show_business_insights(df, models):
    st.title("Business Insights")
    
    st.markdown("""
    ### Key Business Insights and Recommendations
    
    Based on the analysis of customer purchasing behaviors, here are the key insights and actionable recommendations.
    """)
    
    # Age group analysis
    st.subheader("Age Group Analysis")
    
    age_bins = [18, 25, 35, 45, 55, 80]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    
    age_metrics = df.groupby('age_group').agg({
        'purchase_amount': 'mean',
        'loyalty_score': 'mean',
        'purchase_frequency': 'mean',
        'user_id': 'count'
    }).rename(columns={'user_id': 'count'})
    
    fig_age = px.bar(
        age_metrics.reset_index().melt(id_vars='age_group', value_vars=['purchase_amount', 'loyalty_score', 'purchase_frequency']),
        x='age_group',
        y='value',
        color='variable',
        barmode='group',
        title='Key Metrics by Age Group',
        labels={'value': 'Value', 'variable': 'Metric', 'age_group': 'Age Group'},
        color_discrete_sequence=px.colors.qualitative.G10,
        template="plotly_dark"  # Dark theme
    )
    
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Income level analysis
    st.subheader("Income Level Analysis")
    
    income_bins = [0, 40000, 50000, 60000, 70000, 100000]
    income_labels = ['<$40K', '$40K-$50K', '$50K-$60K', '$60K-$70K', '$70K+']
    df['income_level'] = pd.cut(df['annual_income'], bins=income_bins, labels=income_labels)
    
    income_metrics = df.groupby('income_level').agg({
        'purchase_amount': 'mean',
        'loyalty_score': 'mean',
        'purchase_frequency': 'mean',
        'user_id': 'count'
    }).rename(columns={'user_id': 'count'})
    
    fig_income = px.bar(
        income_metrics.reset_index().melt(id_vars='income_level', value_vars=['purchase_amount', 'loyalty_score', 'purchase_frequency']),
        x='income_level',
        y='value',
        color='variable',
        barmode='group',
        title='Key Metrics by Income Level',
        labels={'value': 'Value', 'variable': 'Metric', 'income_level': 'Income Level'},
        color_discrete_sequence=px.colors.qualitative.G10,
        template="plotly_dark"  # Dark theme
    )
    
    st.plotly_chart(fig_income, use_container_width=True)
    
    # Regional performance
    st.subheader("Regional Performance Analysis")
    
    region_metrics = df.groupby('region').agg({
        'purchase_amount': 'sum',
        'user_id': 'count'
    }).rename(columns={'user_id': 'customer_count'})
    
    region_metrics['avg_purchase'] = region_metrics['purchase_amount'] / region_metrics['customer_count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_region_sales = px.pie(
            region_metrics.reset_index(),
            values='purchase_amount',
            names='region',
            title='Total Sales by Region',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_dark"  # Dark theme
        )
        st.plotly_chart(fig_region_sales, use_container_width=True)
    
    with col2:
        fig_region_avg = px.bar(
            region_metrics.reset_index(),
            x='region',
            y='avg_purchase',
            title='Average Purchase by Region',
            color='region',
            text_auto='.2f',
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_dark"  # Dark theme
        )
        st.plotly_chart(fig_region_avg, use_container_width=True)
    
    # Purchase frequency distribution
    st.subheader("Purchase Frequency Analysis")
    
    freq_bins = [0, 10, 15, 20, 25, 30]
    freq_labels = ['Very Low (0-10)', 'Low (11-15)', 'Medium (16-20)', 'High (21-25)', 'Very High (26+)']
    df['frequency_segment'] = pd.cut(df['purchase_frequency'], bins=freq_bins, labels=freq_labels)
    
    freq_metrics = df.groupby('frequency_segment').agg({
        'purchase_amount': 'mean',
        'loyalty_score': 'mean',
        'user_id': 'count'
    }).rename(columns={'user_id': 'count'})
    
    fig_freq = px.bar(
        freq_metrics.reset_index(),
        x='frequency_segment',
        y='count',
        title='Customer Distribution by Purchase Frequency',
        color='frequency_segment',
        text_auto='.0f',
        color_discrete_sequence=px.colors.sequential.Viridis,
        template="plotly_dark"  # Dark theme
    )
    
    st.plotly_chart(fig_freq, use_container_width=True)
    
    # Key business recommendations
    st.subheader("Key Business Recommendations")
    
    recommendations = [
        {
            "title": "Target High-Value Segments",
            "description": "Focus marketing efforts on customers in the highest-value segment (typically older, higher-income customers with high loyalty scores).",
            "expected_impact": "Increase in high-value customer retention and lifetime value."
        },
        {
            "title": "Regional Optimization",
            "description": f"The {df.groupby('region')['purchase_amount'].mean().idxmax()} region shows the highest average purchase amounts. Consider expanding presence or marketing efforts in this region.",
            "expected_impact": "Improved regional sales performance and market penetration."
        },
        {
            "title": "Age-Based Marketing",
            "description": "Develop targeted campaigns for different age segments, with premium offerings for the 46-55 age group which shows highest loyalty.",
            "expected_impact": "Better campaign ROI and improved customer engagement."
        },
        {
            "title": "Loyalty Program Enhancement",
            "description": "Create tiered loyalty programs that reward higher purchase frequency, especially targeting the medium loyalty segment for growth.",
            "expected_impact": "Increased purchase frequency and customer retention."
        },
        {
            "title": "Re-engagement Campaigns",
            "description": "Develop special re-engagement campaigns for low frequency purchasers in the lower loyalty score segments.",
            "expected_impact": "Reduced customer churn and increased purchase frequency."
        }
    ]
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"{i+1}. {rec['title']}"):
            st.write(f"**Description:** {rec['description']}")
            st.write(f"**Expected Impact:** {rec['expected_impact']}")
    
    # Final insights
    st.info("""
    ### Overall Business Insight
    
    The analysis reveals strong correlations between income levels, age, purchase frequency, and loyalty scores.
    By segmenting the customer base and developing targeted strategies for each segment, the business can 
    optimize marketing spend, improve customer retention, and increase overall profitability.
    """)

# Run the app
if __name__ == '__main__':
    main()