import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import json
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== HELPER ==========
def feature_sets(sem):
    if sem == 1:
        return ['age','ug_percentage','ug_cgpa','attendance_percent','internal_score','gender_encoded','course_encoded']
    elif sem == 2:
        return ['age','sem1_sgpa','ug_percentage','ug_cgpa','attendance_percent','internal_score','backlogs','gender_encoded','course_encoded']
    elif sem == 3:
        return ['age','sem1_sgpa','sem2_sgpa','ug_percentage','ug_cgpa','attendance_percent','internal_score','backlogs','gender_encoded','course_encoded']
    else:
        return ['age','sem1_sgpa','sem2_sgpa','sem3_sgpa','ug_percentage','ug_cgpa','attendance_percent','internal_score','backlogs','gender_encoded','course_encoded']

# ========== LOGIN with roles ==========
VALID_USERS = {
    "student": ("student123", "student"),
    "teacher": ("teacher123", "teacher"),
    "admin": ("admin123", "admin"),
}

def login():
    st.markdown(
        """
        <div style='text-align:center; padding:2rem 0 1rem 0;'>
            <div style='font-size:3rem; animation: pulse 2s infinite;'>🎓</div>
            <h1 style='font-family: "Inter", sans-serif; font-weight:700; background: linear-gradient(135deg, #00d4ff, #8b5cf6); -webkit-background-clip: text; background-clip: text; color: transparent;'>Student Performance</h1>
            <p style='color:#8b949e;'>Predictive Intelligence System</p>
        </div>
        <style>
        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.7; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(1); opacity: 0.7; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", placeholder="student / teacher / admin")
        password = st.text_input("Password", type="password", placeholder="••••••")
        if st.button("🔐 Login", use_container_width=True):
            if username in VALID_USERS and VALID_USERS[username][0] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = VALID_USERS[username][1]
                st.rerun()
            else:
                st.error("❌ Access Denied")
    st.stop()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    login()

# ========== CSS (same as polished version) ==========
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1120;
}

.main-header {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(18,20,28,0.95));
    border-bottom: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(120deg, #ffffff, #00d4ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0;
}
.main-header p {
    font-size: 0.8rem;
    color: #8b949e;
    margin: 0.3rem 0 0;
}

.metric-card {
    background: #1e2a3a;
    border-radius: 16px;
    padding: 0.8rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid #2d3a5e;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,212,255,0.1);
    border-color: #00d4ff;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-card .label {
    font-size: 0.7rem;
    color: #8b949e;
    letter-spacing: 0.5px;
}

.pred-card {
    background: #1a2332;
    border-radius: 20px;
    padding: 1rem;
    margin: 0.4rem;
    border: 1px solid #2d3a5e;
    transition: all 0.25s ease;
}
.pred-card:hover {
    transform: translateY(-4px);
    border-color: #00d4ff;
    box-shadow: 0 12px 28px rgba(0,212,255,0.12);
}
.pred-card h4 {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.6rem 0;
    color: #00d4ff;
    text-align: center;
}
.prediction-badge {
    font-size: 1.5rem;
    font-weight: 800;
    padding: 0.4rem;
    border-radius: 14px;
    text-align: center;
    background: #0a0c10;
    margin-bottom: 0.8rem;
}
.prob-bar {
    background: #2d3a5e;
    border-radius: 10px;
    height: 8px;
    margin: 0.3rem 0;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.3s ease;
}

/* Final prediction box */
.final-prediction {
    background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(139,92,246,0.08));
    border: 1px solid #00d4ff;
    border-radius: 28px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,212,255,0.15);
}
.final-prediction h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a0aec0;
    margin: 0 0 0.5rem 0;
    letter-spacing: 1px;
}
.final-badge {
    font-size: 2.5rem;
    font-weight: 800;
    background: #0a0c10;
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 40px;
    margin-top: 0.2rem;
}
.excellent-final { color: #3fb950; border: 1px solid #3fb950; background: rgba(63,185,80,0.1); }
.average-final { color: #e6a817; border: 1px solid #e6a817; background: rgba(230,168,23,0.1); }
.atrisk-final { color: #f85149; border: 1px solid #f85149; background: rgba(248,81,73,0.1); }

.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    margin: 1.5rem 0 0.8rem 0;
    color: #e6a817;
    border-left: 3px solid #e6a817;
    padding-left: 0.8rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    font-weight: 500;
    border-radius: 10px 10px 0 0;
    padding: 0.4rem 1.2rem;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff;
    border-bottom: 2px solid #00d4ff;
    background: rgba(0,212,255,0.05);
}

.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #0f2a4a);
    border: none;
    color: white;
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,212,255,0.2);
    background: linear-gradient(135deg, #2a4a6f, #1a3a5a);
}

.stTextInput > div > div > input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div {
    background: #1a2332;
    border: 1px solid #2d3a5e;
    border-radius: 10px;
    color: white;
}
.stTextInput > div > div > input:focus, .stNumberInput input:focus {
    border-color: #00d4ff;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.1);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #0a0f1c);
    border-right: 1px solid #1e2a3a;
}
[data-testid="stSidebar"] .stMarkdown {
    color: #e6edf3;
}
.sidebar-header {
    text-align: center;
    margin-bottom: 1rem;
}
.sidebar-header h3 {
    color: #00d4ff;
    font-weight: 600;
}
.accuracy-table {
    font-size: 0.8rem;
}

.footer {
    text-align: center;
    margin-top: 2.5rem;
    padding: 1rem;
    color: #6b7280;
    font-size: 0.7rem;
    border-top: 1px solid #2d3a5e;
}
</style>
""",
    unsafe_allow_html=True,
)

# ========== LOAD MODELS ==========
@st.cache_resource
def load_triple_models():
    le_gender = joblib.load("models/le_gender.pkl")
    le_course = joblib.load("models/le_course.pkl")
    models = {}
    for sem in [1,2,3,4]:
        models[f'rf_{sem}'] = joblib.load(f"models/rf_sem{sem}.pkl")
        models[f'xgb_{sem}'] = joblib.load(f"models/xgb_sem{sem}.pkl")
        models[f'svm_{sem}'] = joblib.load(f"models/svm_sem{sem}.pkl")
        models[f'scaler_{sem}'] = joblib.load(f"models/scaler_sem{sem}.pkl")
        models[f'le_target_{sem}'] = joblib.load(f"models/le_target_sem{sem}.pkl")
    return le_gender, le_course, models

@st.cache_data
def load_data():
    dfs = []
    for sem in [1,2,3,4]:
        f = f"student_data_sem{sem}.csv"
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        st.error("❌ No data files found. Run train_triple_models.py first.")
        st.stop()

@st.cache_data
def load_accuracies():
    try:
        with open("models/accuracies_triple.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"Random Forest": {}, "XGBoost": {}, "SVM": {}}

le_gender, le_course, models = load_triple_models()
df = load_data()
accuracies = load_accuracies()

# ========== PREDICTION ==========
def predict_all(semester, data_input):
    features = feature_sets(semester)
    arr = [data_input[f] for f in features]
    X = np.array([arr])
    scaler = models[f'scaler_{semester}']
    X_scaled = scaler.transform(X)
    
    rf = models[f'rf_{semester}']
    rf_pred = rf.predict(X_scaled)[0]
    rf_proba = rf.predict_proba(X_scaled)[0]
    rf_proba_dict = dict(zip(rf.classes_, rf_proba))
    
    xgb = models[f'xgb_{semester}']
    le_target = models[f'le_target_{semester}']
    xgb_pred_enc = xgb.predict(X_scaled)[0]
    xgb_pred = le_target.inverse_transform([xgb_pred_enc])[0]
    xgb_proba = xgb.predict_proba(X_scaled)[0]
    xgb_proba_dict = dict(zip(le_target.classes_, xgb_proba))
    
    svm = models[f'svm_{semester}']
    svm_pred = svm.predict(X_scaled)[0]
    svm_proba = svm.predict_proba(X_scaled)[0]
    svm_proba_dict = dict(zip(svm.classes_, svm_proba))
    
    return {
        "Random Forest": (rf_pred, rf_proba_dict),
        "XGBoost": (xgb_pred, xgb_proba_dict),
        "SVM": (svm_pred, svm_proba_dict)
    }

def get_recommendations(pred_perf, attendance, internal, backlogs, semester):
    rec = []
    if attendance < 75:
        rec.append("🚨 Not eligible for exams – Attendance <75%.")
    elif attendance < 85:
        rec.append("📌 Focus on remaining assignments/tests to improve internals.")
    else:
        rec.append("✅ Good attendance.")
    # Updated threshold: internal < 10 (per subject) triggers low marks warning
    if internal < 10:
        rec.append("📝 Internal marks are low (<10 per subject). Ask for retests or extra assignments.")
    elif internal < 22:
        rec.append("📘 Submit pending assignments to raise internals.")
    else:
        rec.append("🎯 Strong internal marks.")
    if semester >= 2 and backlogs > 0:
        if backlogs >= 5:
            rec.append(f"⚠️ {backlogs} backlogs – apply for clearance courses.")
        else:
            rec.append(f"📚 Clear {backlogs} backlog(s) via makeup exams.")
    if pred_perf == "Excellent":
        rec.append("🏆 Predicted Excellent – continue same effort.")
        rec.append("📈 Aim for SGPA >8.5 next sem.")
    elif pred_perf == "Average":
        rec.append("📌 Predicted Average – increase study hours next semester.")
        rec.append("🎯 Target SGPA 7.5 next sem.")
    else:
        rec.append("🚨 Predicted At‑risk – meet academic advisor urgently.")
        rec.append("⏳ Use tutor support, solve previous papers.")
    if semester == 1:
        rec.append("🔰 Prepare for second semester: maintain attendance.")
    elif semester == 2:
        rec.append("⚙️ Second semester workload increases – plan weekly revision.")
    elif semester == 3:
        rec.append("💼 Start placement preparation.")
    else:
        rec.append("🎓 Final semester – focus on project and clear backlogs.")
    rec = list(dict.fromkeys(rec))
    return rec[:5]

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h3>🎓 EDU PREDICT</h3></div>', unsafe_allow_html=True)
    st.markdown(f"👋 **{st.session_state.username.capitalize()}** (Role: {st.session_state.role.capitalize()})")
    st.markdown("---")
    st.markdown("#### 📈 Model Accuracies")
    acc_df = pd.DataFrame({
        "Sem": ["1", "2", "3", "4"],
        "RF": [accuracies.get("Random Forest", {}).get("Semester 1", "N/A"),
               accuracies.get("Random Forest", {}).get("Semester 2", "N/A"),
               accuracies.get("Random Forest", {}).get("Semester 3", "N/A"),
               accuracies.get("Random Forest", {}).get("Semester 4", "N/A")],
        "XGB": [accuracies.get("XGBoost", {}).get("Semester 1", "N/A"),
                accuracies.get("XGBoost", {}).get("Semester 2", "N/A"),
                accuracies.get("XGBoost", {}).get("Semester 3", "N/A"),
                accuracies.get("XGBoost", {}).get("Semester 4", "N/A")],
        "SVM": [accuracies.get("SVM", {}).get("Semester 1", "N/A"),
                accuracies.get("SVM", {}).get("Semester 2", "N/A"),
                accuracies.get("SVM", {}).get("Semester 3", "N/A"),
                accuracies.get("SVM", {}).get("Semester 4", "N/A")],
    })
    st.dataframe(acc_df, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.markdown("#### 📌 Key Rules")
    st.caption("• Attendance <75% → Not eligible")
    st.caption("• Backlog caps: Sem2≤6, Sem3≤12, Sem4≤18")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# ========== HEADER ==========
st.markdown(
    '<div class="main-header"><h1>🎓 Student Performance Predictor</h1><p>Random Forest vs XGBoost vs SVM | Triple Model System</p></div>',
    unsafe_allow_html=True,
)

# ========== ROLE‑BASED TABS ==========
role = st.session_state.role

if role == "student":
    # Student sees: Predict + About (no Analytics)
    tabs = st.tabs(["📝 Predict", "ℹ️ About"])
    
    with tabs[0]:
        st.markdown("### 📋 Student Details")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name/ID", placeholder="Student name/ID")
            age = st.number_input("Age", 20, 60, 22)
            gender = st.selectbox("Gender", ["Male", "Female"])
            ug_percentage = st.number_input("UG Percentage (%)", 50.0, 100.0, 70.0, step=1.0)
        with col2:
            semester = st.selectbox("Current Semester", [1,2,3,4])
            attendance = st.number_input("Attendance (%)", 0, 100, 80)
            # New: Total internal marks out of 180 (6 subjects × 30)
            total_internal = st.number_input("Total Internal Marks (out of 180)", 0, 180, 120, step=1)
            internal = total_internal / 6.0
            st.caption(f"📊 Per‑subject average internal score: {internal:.1f} / 30")

        ug_cgpa = round(ug_percentage / 9.5, 2)
        if ug_cgpa > 10.0: ug_cgpa = 10.0
        st.caption(f"ℹ️ UG CGPA = {ug_cgpa} (Percentage / 9.5)")

        if semester == 1:
            sem1 = sem2 = sem3 = None
            backlogs = 0
        elif semester == 2:
            sem1 = st.number_input("Sem1 SGPA", 4.0,10.0,7.0,0.1)
            backlogs = st.number_input("Backlogs (max 6)", 0,6,0)
            sem2 = sem3 = None
        elif semester == 3:
            sem1 = st.number_input("Sem1 SGPA", 4.0,10.0,7.0,0.1)
            sem2 = st.number_input("Sem2 SGPA", 4.0,10.0,7.0,0.1)
            backlogs = st.number_input("Backlogs (max 12)", 0,12,0)
            sem3 = None
        else:
            sem1 = st.number_input("Sem1 SGPA", 4.0,10.0,7.0,0.1)
            sem2 = st.number_input("Sem2 SGPA", 4.0,10.0,7.0,0.1)
            sem3 = st.number_input("Sem3 SGPA", 4.0,10.0,7.0,0.1)
            backlogs = st.number_input("Backlogs (max 18)", 0,18,0)

        if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
            if not name:
                st.warning("Enter name")
            elif attendance < 75:
                st.error(f"❌ NOT ELIGIBLE (Attendance {attendance}% <75%)")
            else:
                data = {
                    'age': age, 'ug_percentage': ug_percentage, 'ug_cgpa': ug_cgpa,
                    'attendance_percent': attendance, 'internal_score': internal,
                    'gender_encoded': le_gender.transform([gender])[0],
                    'course_encoded': le_course.transform(["MCA"])[0],
                    'backlogs': backlogs,
                    'sem1_sgpa': sem1 if sem1 is not None else 0.0,
                    'sem2_sgpa': sem2 if sem2 is not None else 0.0,
                    'sem3_sgpa': sem3 if sem3 is not None else 0.0,
                }
                results = predict_all(semester, data)
                
                # Overrides (same as before)
                if semester == 2 and backlogs >= 4:
                    for model in results:
                        results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                    st.warning("⚠️ High backlog count (≥4) → prediction overridden to At‑risk")
                elif semester == 3 and backlogs >= 8:
                    for model in results:
                        results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                    st.warning("⚠️ High backlog count (≥8) → prediction overridden to At‑risk")
                elif semester == 4 and backlogs >= 10:
                    for model in results:
                        results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                    st.warning("⚠️ High backlog count (≥10) → prediction overridden to At‑risk")
                
                if semester >= 2 and backlogs >= 3:
                    if semester == 2 and sem1 is not None and sem1 > 8.0:
                        for model in results:
                            results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                        st.warning("⚠️ Unrealistic: high backlogs + high SGPA → overridden to At‑risk")
                    elif semester == 3 and sem2 is not None and sem2 > 8.0:
                        for model in results:
                            results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                        st.warning("⚠️ Unrealistic: high backlogs + high SGPA → overridden to At‑risk")
                    elif semester == 4 and sem3 is not None and sem3 > 8.0:
                        for model in results:
                            results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                        st.warning("⚠️ Unrealistic: high backlogs + high SGPA → overridden to At‑risk")
                
                predictions = [results[m][0] for m in ["Random Forest", "XGBoost", "SVM"]]
                final_pred = max(set(predictions), key=predictions.count)
                final_class = "excellent-final" if final_pred == "Excellent" else "average-final" if final_pred == "Average" else "atrisk-final"
                
                st.subheader(f"📊 Results for {name}")
                st.markdown(f"""
                <div class="final-prediction">
                    <h3>🎯 FINAL PREDICTION (Majority Vote)</h3>
                    <div class="final-badge {final_class}" style="font-size:2.2rem;">{final_pred}</div>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(3)
                for i, (model_name, (pred, probs)) in enumerate(results.items()):
                    with cols[i]:
                        st.markdown(f"<div class='pred-card'><h4>{model_name}</h4>", unsafe_allow_html=True)
                        bg_color = "#0d3123" if pred=="Excellent" else "#2d2208" if pred=="Average" else "#2d1010"
                        text_color = "#3fb950" if pred=="Excellent" else "#e6a817" if pred=="Average" else "#f85149"
                        st.markdown(f"<div class='prediction-badge' style='background:{bg_color}; color:{text_color};'>{pred}</div>", unsafe_allow_html=True)
                        for k, v in probs.items():
                            col = '#3fb950' if k=='Excellent' else '#e6a817' if k=='Average' else '#f85149'
                            st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:0.8rem;'><span>{k}</span><span>{v:.1%}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{v*100}%; background:{col};'></div></div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                rec = get_recommendations(final_pred, attendance, internal, backlogs, semester)
                st.markdown("#### 💡 Recommendations")
                for r in rec:
                    st.markdown(f"- {r}")
    
    with tabs[1]:
        # About tab content
        st.markdown("### ℹ️ About")
        st.markdown("""
**Student Performance Predictor** – MCA Mini Project  
- **Three algorithms**: Random Forest, XGBoost, SVM  
- **Dataset**: 4,000 synthetic students (1,000 per semester)  
- **Backlog caps**: Sem2≤6, Sem3≤12, Sem4≤18  
- **Attendance <75%** → Not eligible for exams  
- **Tech**: Python, Streamlit, Scikit‑learn, XGBoost, Plotly  

**Team:** Raghavendra & Hitesh S  
**Guide:** Dr. Hanumanthappa M  
**Institution:** Bangalore University
""")

else:  # teacher or admin
    # Teacher/Admin sees: Predict, Analytics, About
    tabs = st.tabs(["📝 Predict", "📊 Analytics", "ℹ️ About"])
    
    with tabs[0]:
        # Predict tab (same as student version with total internal)
        st.markdown("### 📋 Student Details")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", placeholder="Student name")
            age = st.number_input("Age", 18, 60, 22)
            gender = st.selectbox("Gender", ["Male", "Female"])
            ug_percentage = st.number_input("UG Percentage (%)", 50.0, 100.0, 70.0, step=1.0)
        with col2:
            semester = st.selectbox("Current Semester", [1,2,3,4])
            attendance = st.number_input("Attendance (%)", 0, 100, 80)
            total_internal = st.number_input("Total Internal Marks (out of 180)", 0, 180, 120, step=1)
            internal = total_internal / 6.0
            st.caption(f"📊 Per‑subject average internal score: {internal:.1f} / 30")

        ug_cgpa = round(ug_percentage / 9.5, 2)
        if ug_cgpa > 10.0: ug_cgpa = 10.0
        st.caption(f"ℹ️ UG CGPA = {ug_cgpa} (Percentage / 9.5)")

        if semester == 1:
            sem1 = sem2 = sem3 = None
            backlogs = 0
        elif semester == 2:
            sem1 = st.number_input("Sem1 SGPA", 4.0,10.0,7.0,0.1)
            backlogs = st.number_input("Backlogs (max 6)", 0,6,0)
            sem2 = sem3 = None
        elif semester == 3:
            sem1 = st.number_input("Sem1 SGPA", 4.0,10.0,7.0,0.1)
            sem2 = st.number_input("Sem2 SGPA", 4.0,10.0,7.0,0.1)
            backlogs = st.number_input("Backlogs (max 12)", 0,12,0)
            sem3 = None
        else:
            sem1 = st.number_input("Sem1 SGPA", 4.0,10.0,7.0,0.1)
            sem2 = st.number_input("Sem2 SGPA", 4.0,10.0,7.0,0.1)
            sem3 = st.number_input("Sem3 SGPA", 4.0,10.0,7.0,0.1)
            backlogs = st.number_input("Backlogs (max 18)", 0,18,0)

        if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
            if not name:
                st.warning("Enter name")
            elif attendance < 75:
                st.error(f"❌ NOT ELIGIBLE (Attendance {attendance}% <75%)")
            else:
                data = {
                    'age': age, 'ug_percentage': ug_percentage, 'ug_cgpa': ug_cgpa,
                    'attendance_percent': attendance, 'internal_score': internal,
                    'gender_encoded': le_gender.transform([gender])[0],
                    'course_encoded': le_course.transform(["MCA"])[0],
                    'backlogs': backlogs,
                    'sem1_sgpa': sem1 if sem1 is not None else 0.0,
                    'sem2_sgpa': sem2 if sem2 is not None else 0.0,
                    'sem3_sgpa': sem3 if sem3 is not None else 0.0,
                }
                results = predict_all(semester, data)
                
                # Overrides (same as before)
                if semester == 2 and backlogs >= 4:
                    for model in results:
                        results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                    st.warning("⚠️ High backlog count (≥4) → prediction overridden to At‑risk")
                elif semester == 3 and backlogs >= 8:
                    for model in results:
                        results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                    st.warning("⚠️ High backlog count (≥8) → prediction overridden to At‑risk")
                elif semester == 4 and backlogs >= 10:
                    for model in results:
                        results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                    st.warning("⚠️ High backlog count (≥10) → prediction overridden to At‑risk")
                
                if semester >= 2 and backlogs >= 3:
                    if semester == 2 and sem1 is not None and sem1 > 8.0:
                        for model in results:
                            results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                        st.warning("⚠️ Unrealistic: high backlogs + high SGPA → overridden to At‑risk")
                    elif semester == 3 and sem2 is not None and sem2 > 8.0:
                        for model in results:
                            results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                        st.warning("⚠️ Unrealistic: high backlogs + high SGPA → overridden to At‑risk")
                    elif semester == 4 and sem3 is not None and sem3 > 8.0:
                        for model in results:
                            results[model] = ("At-risk", {k: (1.0 if k=="At-risk" else 0.0) for k in ["Excellent","Average","At-risk"]})
                        st.warning("⚠️ Unrealistic: high backlogs + high SGPA → overridden to At‑risk")
                
                predictions = [results[m][0] for m in ["Random Forest", "XGBoost", "SVM"]]
                final_pred = max(set(predictions), key=predictions.count)
                final_class = "excellent-final" if final_pred == "Excellent" else "average-final" if final_pred == "Average" else "atrisk-final"
                
                st.subheader(f"📊 Results for {name}")
                st.markdown(f"""
                <div class="final-prediction">
                    <h3>🎯 FINAL PREDICTION (Majority Vote)</h3>
                    <div class="final-badge {final_class}" style="font-size:2.2rem;">{final_pred}</div>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(3)
                for i, (model_name, (pred, probs)) in enumerate(results.items()):
                    with cols[i]:
                        st.markdown(f"<div class='pred-card'><h4>{model_name}</h4>", unsafe_allow_html=True)
                        bg_color = "#0d3123" if pred=="Excellent" else "#2d2208" if pred=="Average" else "#2d1010"
                        text_color = "#3fb950" if pred=="Excellent" else "#e6a817" if pred=="Average" else "#f85149"
                        st.markdown(f"<div class='prediction-badge' style='background:{bg_color}; color:{text_color};'>{pred}</div>", unsafe_allow_html=True)
                        for k, v in probs.items():
                            col = '#3fb950' if k=='Excellent' else '#e6a817' if k=='Average' else '#f85149'
                            st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:0.8rem;'><span>{k}</span><span>{v:.1%}</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{v*100}%; background:{col};'></div></div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                rec = get_recommendations(final_pred, attendance, internal, backlogs, semester)
                st.markdown("#### 💡 Recommendations")
                for r in rec:
                    st.markdown(f"- {r}")
    
    with tabs[1]:
        # Analytics tab (unchanged)
        st.markdown("### 📊 Analytics Dashboard")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown('<div class="metric-card"><div class="value">'+str(len(df))+'</div><div class="label">📊 Total Students</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="value">{df["attendance_percent"].mean():.1f}%</div><div class="label">📅 Avg Attendance</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="value">{len(df[df["backlogs"]>0])}</div><div class="label">📚 With Backlogs</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="value">{len(df[df["attendance_percent"]<75])}</div><div class="label">⚠️ Not Eligible</div></div>', unsafe_allow_html=True)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Performance Distribution")
            perf_counts = df["performance"].value_counts()
            fig = px.pie(values=perf_counts.values, names=perf_counts.index,
                         color_discrete_map={"Excellent":"#00e5b2","Average":"#f5b042","At-risk":"#ff5e7e"}, hole=0.3)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0", margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col_chart2:
            st.subheader("Attendance vs Performance")
            df["att_range"] = pd.cut(df["attendance_percent"], bins=[0,60,75,85,101], labels=["<60%","60-75%","75-85%","≥85%"])
            att_counts = df["att_range"].value_counts().sort_index()
            colors = ["#ff5e7e" if l in ["<60%","60-75%"] else "#00e5b2" for l in att_counts.index]
            fig2 = px.bar(x=att_counts.index, y=att_counts.values, color=att_counts.index, color_discrete_sequence=colors,
                          title="Attendance Range")
            fig2.update_layout(xaxis_title="Attendance", yaxis_title="Count", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🧠 Feature Importance")
        model_choice = st.radio("Select Model", ["Random Forest", "XGBoost", "SVM"], horizontal=True)
        semester_choice = st.selectbox("Semester", ["Semester 1","Semester 2","Semester 3","Semester 4"])
        sem_idx = {"Semester 1":1,"Semester 2":2,"Semester 3":3,"Semester 4":4}[semester_choice]

        if sem_idx == 1: feature_names = ["Age","UG%","UG CGPA","Attendance%","Internal","Gender","Course"]
        elif sem_idx == 2: feature_names = ["Age","Sem1","UG%","UG CGPA","Attendance%","Internal","Backlogs","Gender","Course"]
        elif sem_idx == 3: feature_names = ["Age","Sem1","Sem2","UG%","UG CGPA","Attendance%","Internal","Backlogs","Gender","Course"]
        else: feature_names = ["Age","Sem1","Sem2","Sem3","UG%","UG CGPA","Attendance%","Internal","Backlogs","Gender","Course"]

        if model_choice == "Random Forest":
            model = models[f'rf_{sem_idx}']
            importance = model.feature_importances_
        elif model_choice == "XGBoost":
            model = models[f'xgb_{sem_idx}']
            importance = model.feature_importances_
        else:
            data_sem = pd.read_csv(f'student_data_sem{sem_idx}.csv')
            data_sem['gender_encoded'] = le_gender.transform(data_sem['gender'])
            data_sem['course_encoded'] = le_course.transform(data_sem['course'])
            X_sem = data_sem[feature_sets(sem_idx)]
            y_sem = data_sem['performance']
            sample = min(300, len(X_sem))
            X_sample, y_sample = X_sem.sample(n=sample, random_state=42), y_sem.sample(n=sample, random_state=42)
            scaler = models[f'scaler_{sem_idx}']
            X_scaled = scaler.transform(X_sample)
            svm = models[f'svm_{sem_idx}']
            perm = permutation_importance(svm, X_scaled, y_sample, n_repeats=3, random_state=42, n_jobs=-1)
            importance = perm.importances_mean

        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values("Importance", ascending=True)
        fig3 = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Bluered",
                     title=f"{model_choice} – {semester_choice}")
        fig3.update_layout(height=450, yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0")
        st.plotly_chart(fig3, use_container_width=True)
    
    with tabs[2]:
        # About tab
        st.markdown("### ℹ️ About")
        st.markdown("""
**Student Performance Predictor** – MCA Mini Project  
- **Three algorithms**: Random Forest, XGBoost, SVM  
- **Dataset**: 4,000 synthetic students (1,000 per semester)  
- **Backlog caps**: Sem2≤6, Sem3≤12, Sem4≤18  
- **Attendance <75%** → Not eligible for exams  
- **Tech**: Python, Streamlit, Scikit‑learn, XGBoost, Plotly  

**Team:** Raghavendra & Hitesh S  
**Guide:** Dr. Hanumanthappa M  
**Institution:** Bangalore University
""")

# ========== FOOTER ==========
st.markdown('<div class="footer">🎓 Student Performance Prediction System | Triple Model (RF + XGB + SVM)</div>', unsafe_allow_html=True)