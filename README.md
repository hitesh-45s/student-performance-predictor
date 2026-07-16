# 🎓 Student Performance Prediction Pipeline

## 📌 Project Overview
An end-to-end machine learning system designed to forecast semester-wise academic outcomes. This project utilizes a Triple-Model architecture (Random Forest, XGBoost, SVM) to analyze academic metrics and predict whether a student's future performance will be "Excellent," "Average," or "At-risk." 

## 🛠️ Technology Stack
*   **Languages:** Python
*   **Machine Learning:** Scikit-Learn (Random Forest, SVM), XGBoost
*   **Data Processing:** Pandas, NumPy, Synthetic Data Generation (Faker)
*   **Frontend UI & Analytics:** Streamlit, Plotly

## 🚀 Key Features
*   **Triple-Model Ensemble:** Compares predictions across three separate algorithms for maximum accuracy (achieving up to 93% accuracy).
*   **Role-Based Access:** Custom views for Students, Teachers, and Admins.
*   **Analytics Dashboard:** Visualizes feature importance, performance distributions, and attendance impacts using Plotly.
*   **Dynamic Rules Engine:** Automatically flags students with high backlogs or low attendance (<75%).

## 🤝 Team & Contributions
This was a collaborative academic project developed by a two-person team (Hitesh & Raghavendra). 

**My specific technical contributions included:**

*  Leading system testing and Quality Assurance (QA) for the frontend Streamlit application.

*  Conducting business logic validation to identify edge cases, out-of-bounds parameters, and logical flaws in model outputs.

*  Defining the real-world academic constraints used to refine prediction reliability and trigger system warnings for unrealistic data inputs.

## ⚙️ How to Run Locally
1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`
