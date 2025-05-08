\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{titlesec}
\usepackage{xcolor}

\titleformat{\section}{\large\bfseries\color{blue}}{\thesection}{1em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{black}}{\thesubsection}{1em}{}

\title{\textbf{Customer Purchasing Behavior Analysis Dashboard}}
\author{}
\date{}

\begin{document}

\maketitle

\section*{📊 Overview}
This project is an interactive \textbf{Streamlit dashboard} designed to analyze and visualize customer purchasing behavior. It allows businesses to:
\begin{itemize}
  \item Explore customer data interactively.
  \item Segment users using K-Means clustering.
  \item Predict loyalty scores using machine learning.
  \item Derive actionable business insights.
\end{itemize}

\section*{🚀 Problem Statement}
The goal is to help businesses understand customer patterns to improve marketing strategies and retention by analyzing demographics, purchase behavior, and loyalty factors.

\section*{🧠 Key Features}
\subsection*{Data Exploration}
\begin{itemize}
  \item Descriptive stats, histograms, and scatter plots.
  \item Correlation matrix and regional breakdowns.
\end{itemize}

\subsection*{Customer Segmentation}
\begin{itemize}
  \item K-Means clustering with 3D visualization.
  \item Cluster profiling and radar comparisons.
  \item Export segmented data to CSV.
\end{itemize}

\subsection*{Loyalty Score Prediction}
\begin{itemize}
  \item Predict loyalty based on input.
  \item Random Forest regression with feature importance.
  \item Business interpretation of predictions.
\end{itemize}

\subsection*{Business Insights}
\begin{itemize}
  \item Segment performance by age, income, region, and frequency.
  \item Visual dashboards and charts.
  \item Strategic marketing recommendations.
\end{itemize}

\section*{🛠️ Technology Stack}
Python, Streamlit, scikit-learn, Pandas, Plotly, Joblib

\section*{📂 Project Structure}
\begin{verbatim}
.
├── app.py
├── Customer_Purchasing_Behavior.ipynb
├── kmeans_model.pkl
├── rf_model.pkl
├── scaler.pkl
├── Customer_Purchasing_Behaviors.csv
\end{verbatim}

\section*{🔧 Running Instructions}
\begin{enumerate}
  \item Clone the repository:
  \begin{verbatim}
  git clone https://github.com/yourusername/customer-behavior-analysis.git
  cd customer-behavior-analysis
  \end{verbatim}
  \item Install dependencies:
  \begin{verbatim}
  pip install -r requirements.txt
  \end{verbatim}
  \item Run the app:
  \begin{verbatim}
  streamlit run app.py
  \end{verbatim}
\end{enumerate}

\section*{💡 Sample Insights}
\begin{itemize}
  \item Customers aged 46–55 and earning \$70K+ are highly loyal.
  \item The West region has the highest average purchase value.
  \item Loyalty increases with frequent purchases.
\end{itemize}

\section*{✅ Future Enhancements}
\begin{itemize}
  \item Cloud integration for datasets/models.
  \item Time-series trend tracking.
  \item Session-based personalization.
\end{itemize}

\section*{❤️ Built with Open-Source Tools}
\begin{itemize}
  \item Streamlit
  \item scikit-learn
  \item Plotly
\end{itemize}

\end{document}
