<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Customer Purchasing Behavior Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; background: #f9f9f9; color: #333; }
    h1, h2 { color: #0073e6; }
    code, pre { background: #eee; padding: 4px; border-radius: 4px; }
  </style>
</head>
<body>

<h1>📊 Customer Purchasing Behavior Analysis Dashboard</h1>

<p><strong>Streamlit-based interactive dashboard</strong> to analyze customer purchase patterns, segment users, predict loyalty, and generate business insights.</p>

<h2>🚀 Problem Statement</h2>
<p>This dashboard helps businesses understand their customer base by analyzing demographics, purchasing patterns, and predicting loyalty scores using machine learning.</p>

<h2>🧠 Key Features</h2>
<h3>🔍 Data Exploration</h3>
<ul>
  <li>Histograms, scatter plots, and correlation matrix</li>
  <li>Region-wise breakdown of metrics</li>
</ul>

<h3>🎯 Customer Segmentation</h3>
<ul>
  <li>K-Means clustering with 3D visualizations</li>
  <li>Cluster profiling and radar comparisons</li>
  <li>Downloadable segmented CSV</li>
</ul>

<h3>📈 Loyalty Score Prediction</h3>
<ul>
  <li>Predict scores using Random Forest</li>
  <li>Interpret results with feature importance charts</li>
</ul>

<h3>💡 Business Insights</h3>
<ul>
  <li>Insights by age, region, income, frequency</li>
  <li>Strategic marketing recommendations</li>
</ul>

<h2>🛠️ Tech Stack</h2>
<p>Python, Streamlit, Pandas, scikit-learn, Plotly, Joblib</p>

<h2>📂 Project Structure</h2>
<pre>
.
├── app.py
├── Customer_Purchasing_Behavior.ipynb
├── kmeans_model.pkl
├── rf_model.pkl
├── scaler.pkl
├── Customer_Purchasing_Behaviors.csv
</pre>

<h2>📦 How to Run</h2>
<ol>
  <li>Clone the repo:<br>
    <code>git clone https://github.com/yourusername/customer-behavior-analysis.git</code>
  </li>
  <li>Install dependencies:<br>
    <code>pip install -r requirements.txt</code>
  </li>
  <li>Run the app:<br>
    <code>streamlit run app.py</code>
  </li>
</ol>

<h2>💡 Sample Insights</h2>
<ul>
  <li>46–55 year olds with $70K+ income show highest loyalty</li>
  <li>West region performs best in sales</li>
  <li>Loyalty increases with purchase frequency</li>
</ul>

<h2>✅ Future Plans</h2>
<ul>
  <li>Cloud-hosted model/dataset support</li>
  <li>Session-based personalization</li>
</ul>

<h2>❤️ Built with Open-Source Tools</h2>
<p>Streamlit, Plotly, scikit-learn, and more</p>

</body>
</html>
