"""
train_triple_models.py
- Generates semester-specific CSVs (1000 students each)
- Trains Random Forest, XGBoost, and SVM for each semester.
- Saves models, scalers, target encoders, and accuracies.
"""

import pandas as pd
import numpy as np
from faker import Faker
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# CONFIG
STUDENTS_PER_SEM = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
fake = Faker('en_IN')
fake.seed_instance(RANDOM_SEED)

# ---------- Helper functions ----------
def backlog_penalty(backlogs):
    if backlogs == 0: return 1.0
    elif backlogs == 1: return 0.95
    elif backlogs == 2: return 0.85
    elif backlogs == 3: return 0.70
    elif backlogs == 4: return 0.55
    elif backlogs == 5: return 0.40
    elif backlogs == 6: return 0.30
    elif backlogs == 7: return 0.20
    else: return 0.10

def extreme_backlog_risk(backlogs, semester):
    if semester >= 3 and backlogs >= 12:
        return True
    return False

def get_performance(score):
    if score >= 0.80: return "Excellent"
    elif score >= 0.55: return "Average"
    else: return "At-risk"

# ---------- Student generation (exactly as in final.py) ----------
def generate_student(semester, idx):
    student_id = f"SEM{semester}_{idx:04d}"
    name = f"{fake.first_name()} {fake.last_name()}"
    age = np.random.randint(21, 29)
    gender = np.random.choice(['Male', 'Female'])
    course = "MCA"

    # UG scores
    ug_percentage = np.random.normal(72, 10)
    ug_percentage = np.clip(ug_percentage, 45, 95)
    ug_cgpa = (ug_percentage / 10) + np.random.normal(0, 0.35)
    ug_cgpa = np.clip(ug_cgpa, 4.5, 9.8)

    # Current semester features
    attendance = np.random.normal(80, 10)
    attendance = np.clip(attendance, 45, 100)
    internal = np.random.normal(20, 5)
    internal = np.clip(internal, 0, 30)

    # Generate SGPA for previous semesters
    sem1_sgpa = sem2_sgpa = sem3_sgpa = None
    if semester >= 2:
        sem1_sgpa = 4.8 + (attendance/100)*0.6 + (internal/30)*2.5 + (ug_percentage/100)*1.4 + np.random.normal(0, 0.6)
        sem1_sgpa = np.clip(sem1_sgpa, 4.0, 9.8)
    if semester >= 3:
        trend = np.random.choice([-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5],
                                 p=[0.05,0.10,0.15,0.40,0.15,0.10,0.05])
        sem2_sgpa = sem1_sgpa + trend + np.random.normal(0, 0.35)
        sem2_sgpa = np.clip(sem2_sgpa, 4.0, 9.8)
    if semester >= 4:
        trend = np.random.choice([-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5],
                                 p=[0.05,0.10,0.15,0.40,0.15,0.10,0.05])
        sem3_sgpa = sem2_sgpa + trend + np.random.normal(0, 0.35)
        sem3_sgpa = np.clip(sem3_sgpa, 4.0, 9.8)

    # -------- BACKLOG GENERATION (with semester caps) --------
    if semester == 1:
        backlogs = 0
    else:
        risk_score = 0
        if internal < 15: risk_score += 2
        if semester >= 2 and sem1_sgpa < 6: risk_score += 2
        if semester >= 3 and sem2_sgpa < 6: risk_score += 2
        if attendance < 70: risk_score += 1

        if semester == 2:
            max_back = 6
        elif semester == 3:
            max_back = 12
        else:
            max_back = 18

        if risk_score >= 6:
            backlogs = np.random.randint(4, min(13, max_back+1))
        elif risk_score >= 4:
            backlogs = np.random.randint(2, min(8, max_back+1))
        elif risk_score >= 2:
            backlogs = np.random.randint(0, min(5, max_back+1))
        else:
            choices = [0,1,2,3]
            probs = [0.60,0.25,0.10,0.05]
            valid = [(c,p) for c,p in zip(choices,probs) if c <= max_back]
            if valid:
                vals, pvals = zip(*valid)
                pvals = np.array(pvals) / sum(pvals)
                backlogs = np.random.choice(vals, p=pvals)
            else:
                backlogs = 0
        backlogs = min(backlogs, max_back)

    # Extreme backlog risk (only for sem3-4)
    if extreme_backlog_risk(backlogs, semester):
        performance = "At-risk"
    else:
        if semester == 1:
            score = (internal/30)*0.45 + (attendance/100)*0.15 + (ug_percentage/100)*0.20 + (ug_cgpa/10)*0.20
        elif semester == 2:
            score = (sem1_sgpa/10)*0.45 + (internal/30)*0.30 + (attendance/100)*0.10 + (ug_percentage/100)*0.10 + (ug_cgpa/10)*0.05
            score *= backlog_penalty(backlogs)
        elif semester == 3:
            score = (sem2_sgpa/10)*0.50 + (internal/30)*0.28 + (attendance/100)*0.08 + (sem1_sgpa/10)*0.14
            score *= backlog_penalty(backlogs)
        else:
            score = (sem3_sgpa/10)*0.45 + (internal/30)*0.22 + (sem2_sgpa/10)*0.15 + (sem1_sgpa/10)*0.10 + (attendance/100)*0.08
            score *= backlog_penalty(backlogs)

        score += np.random.normal(0, 0.02)
        score = np.clip(score, 0, 1)
        performance = get_performance(score)

    # Build record
    record = {
        'student_id': student_id,
        'student_name': name,
        'age': age,
        'gender': gender,
        'course': course,
        'semester': semester,
        'ug_percentage': round(ug_percentage, 2),
        'ug_cgpa': round(ug_cgpa, 2),
        'attendance_percent': round(attendance, 1),
        'internal_score': int(round(internal)),
        'backlogs': backlogs,
        'performance': performance
    }
    if semester >= 2:
        record['sem1_sgpa'] = round(sem1_sgpa, 2)
    if semester >= 3:
        record['sem2_sgpa'] = round(sem2_sgpa, 2)
    if semester >= 4:
        record['sem3_sgpa'] = round(sem3_sgpa, 2)
    return record

# ---------- Generate CSV files ----------
os.makedirs('data', exist_ok=True)
for sem in [1,2,3,4]:
    students = [generate_student(sem, i) for i in range(1, STUDENTS_PER_SEM+1)]
    df_sem = pd.DataFrame(students)
    csv_name = f'student_data_sem{sem}.csv'
    df_sem.to_csv(csv_name, index=False)
    print(f"Generated {csv_name} with {len(df_sem)} students")
    print(df_sem['performance'].value_counts())
    print("---")

# ---------- Encode categorical features ----------
le_gender = LabelEncoder()
le_course = LabelEncoder()
all_combined = pd.concat([pd.read_csv(f'student_data_sem{sem}.csv') for sem in [1,2,3,4]], ignore_index=True)
le_gender.fit(all_combined['gender'])
le_course.fit(all_combined['course'])

os.makedirs('models', exist_ok=True)

# ---------- Feature sets per semester ----------
feature_sets = {
    1: ['age','ug_percentage','ug_cgpa','attendance_percent','internal_score','gender_encoded','course_encoded'],
    2: ['age','sem1_sgpa','ug_percentage','ug_cgpa','attendance_percent','internal_score','backlogs','gender_encoded','course_encoded'],
    3: ['age','sem1_sgpa','sem2_sgpa','ug_percentage','ug_cgpa','attendance_percent','internal_score','backlogs','gender_encoded','course_encoded'],
    4: ['age','sem1_sgpa','sem2_sgpa','sem3_sgpa','ug_percentage','ug_cgpa','attendance_percent','internal_score','backlogs','gender_encoded','course_encoded']
}

accuracies = {"Random Forest": {}, "XGBoost": {}, "SVM": {}}

def train_models_for_semester(sem):
    df = pd.read_csv(f'student_data_sem{sem}.csv')
    df['gender_encoded'] = le_gender.transform(df['gender'])
    df['course_encoded'] = le_course.transform(df['course'])
    
    features = feature_sets[sem]
    X = df[features]
    y = df['performance']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ----- Random Forest -----
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=RANDOM_SEED)
    rf.fit(X_train_scaled, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
    
    # ----- XGBoost -----
    le_target = LabelEncoder()
    y_train_enc = le_target.fit_transform(y_train)
    y_test_enc = le_target.transform(y_test)
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, eval_metric='mlogloss', use_label_encoder=False)
    xgb_clf.fit(X_train_scaled, y_train_enc)
    xgb_pred = le_target.inverse_transform(xgb_clf.predict(X_test_scaled))
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    # ----- SVM (with probability=True) -----
    svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
    svm.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    
    print(f"Sem {sem} | RF: {rf_acc:.2%} | XGB: {xgb_acc:.2%} | SVM: {svm_acc:.2%}")
    
    # Save models and scaler
    joblib.dump(rf, f'models/rf_sem{sem}.pkl')
    joblib.dump(xgb_clf, f'models/xgb_sem{sem}.pkl')
    joblib.dump(svm, f'models/svm_sem{sem}.pkl')
    joblib.dump(scaler, f'models/scaler_sem{sem}.pkl')
    joblib.dump(le_target, f'models/le_target_sem{sem}.pkl')
    
    accuracies["Random Forest"][f"Semester {sem}"] = f"{rf_acc:.1%}"
    accuracies["XGBoost"][f"Semester {sem}"] = f"{xgb_acc:.1%}"
    accuracies["SVM"][f"Semester {sem}"] = f"{svm_acc:.1%}"
    
   
# Train for all semesters
for sem in [1,2,3,4]:
    train_models_for_semester(sem)

# Save accuracies JSON
with open("models/accuracies_triple.json", "w") as f:
    json.dump(accuracies, f, indent=2)

# Save shared encoders
joblib.dump(le_gender, 'models/le_gender.pkl')
joblib.dump(le_course, 'models/le_course.pkl')


print("\n✅ All models and datasets saved (RF, XGB, SVM).")