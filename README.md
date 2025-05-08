# Customer Purchasing Behavior Analysis

This project focuses on analyzing customer purchasing behaviors, segmenting customers based on their characteristics and habits, predicting customer loyalty, and deriving actionable business insights.

## Problem Statement

The core objective of this analysis is to identify distinct customer segments based on purchasing behaviors and demographics, predict customer loyalty scores, and develop actionable insights to improve marketing strategies and increase customer retention. The business goal is to understand which factors most influence purchase amounts and loyalty, and develop targeted marketing campaigns for different customer segments.

## Project Description

This repository contains the code and resources for a customer purchasing behavior analysis project. The analysis involves:

1. **Exploratory Data Analysis (EDA):** Understanding the dataset, including distributions of key variables (age, income, purchase amount, loyalty score, purchase frequency) and relationships between them (e.g., correlation heatmap, scatter plots). Regional analysis is also performed.

2. **Customer Segmentation:** Applying the K-Means clustering algorithm to group customers into distinct segments based on their features. The characteristics of each segment are then analyzed.

3. **Loyalty Prediction:** Building a predictive model (Random Forest Regressor) to estimate customer loyalty scores based on their attributes.

4. **Business Insights:** Providing actionable recommendations derived from the analysis, aimed at improving marketing strategies and customer retention.

A Streamlit web application is included to provide an interactive dashboard for exploring the data, viewing segmentation results, predicting loyalty scores, and accessing business insights.

## Features

The Streamlit application provides the following sections:

* **Home:** An overview of the project and key dataset metrics.
* **Data Exploration:** Interactive visualizations to explore data distributions, regional differences, and variable relationships.
* **Customer Segmentation:** Displays cluster profiles and visualizations (3D scatter plot, radar chart, region distribution) to understand customer segments.
* **Loyalty Prediction:** An interactive form to input customer information and predict their loyalty score.
* **Business Insights:** Presents actionable recommendations based on the analysis findings.

## Technologies Used

* Python
* Pandas (for data manipulation and analysis)
* NumPy (for numerical operations)
* Matplotlib (for basic plotting)
* Seaborn (for enhanced visualizations)
* Scikit-learn (for clustering and regression models)
* Joblib (for saving and loading models)
* Streamlit (for building the web application)
* Plotly (for interactive visualizations in the web app)

## How to Run

1. Clone the repository (or ensure you have the `Customer_Purchasing_Behaviors.csv`, `Customer_Purchasing_Behaviour.ipynb`, and `app.py` files).

2. Make sure you have the necessary Python libraries installed. You can install them using pip:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit plotly
   ```

3. Run the Jupyter Notebook (`Customer_Purchasing_Behaviour.ipynb`) to perform the analysis, generate visualizations, train the models, and save the `kmeans_model.pkl`, `rf_model.pkl`, and `scaler.pkl` files. These files are required by the Streamlit app.

4. Run the Streamlit application from your terminal:
   ```
   streamlit run app.py
   ```

5. The application will open in your web browser.

## Deployment

You can access the deployed version of the Streamlit application here:
[Insert Deployment Link Here]

## Screenshots

[Include screenshots of the Streamlit application here]

