"""
AI Based Disease Prediction System
Fully self-contained Streamlit app — dataset embedded, no external files.
Author: Ankita (238330028)
Run: streamlit run app.py
"""

import random
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

random.seed(42)

# ==================================================================
# 1. EMBEDDED DATASET
# ==================================================================
RAW_DATA = [
    # ---------- Common Cold & Flu ----------
    ("Common Cold",       ["runny nose", "sore throat", "sneezing", "mild fever", "cough", "fatigue"]),
    ("Common Cold",       ["runny nose", "sneezing", "watery eyes", "sore throat", "mild fever"]),
    ("Common Cold",       ["cough", "runny nose", "sore throat", "fatigue", "sneezing"]),
    ("Influenza",         ["high fever", "body ache", "chills", "fatigue", "headache", "cough"]),
    ("Influenza",         ["high fever", "muscle pain", "headache", "sore throat", "fatigue"]),
    ("Influenza",         ["chills", "body ache", "high fever", "cough", "fatigue"]),

    # ---------- Respiratory ----------
    ("Allergic Rhinitis", ["sneezing", "runny nose", "itchy eyes", "watery eyes", "nasal congestion"]),
    ("Allergic Rhinitis", ["sneezing", "itchy eyes", "runny nose", "watery eyes"]),
    ("Asthma",            ["shortness of breath", "wheezing", "chest tightness", "cough", "difficulty breathing"]),
    ("Asthma",            ["wheezing", "cough", "shortness of breath", "chest tightness"]),
    ("Bronchitis",        ["persistent cough", "chest discomfort", "wheezing", "mild fever", "fatigue"]),
    ("Bronchitis",        ["cough", "chest discomfort", "sore throat", "fatigue", "wheezing"]),
    ("Pneumonia",         ["high fever", "chest pain", "difficulty breathing", "cough", "fatigue", "chills"]),
    ("Pneumonia",         ["high fever", "cough", "chest pain", "shortness of breath", "chills"]),
    ("Tuberculosis",      ["persistent cough", "weight loss", "night sweats", "fever", "loss of appetite", "fatigue"]),
    ("Tuberculosis",      ["cough with blood", "night sweats", "weight loss", "fever", "fatigue"]),

    # ---------- Gastrointestinal ----------
    ("Gastroenteritis",   ["diarrhea", "vomiting", "abdominal pain", "nausea", "mild fever"]),
    ("Gastroenteritis",   ["vomiting", "diarrhea", "stomach cramps", "nausea", "fatigue"]),
    ("Food Poisoning",    ["nausea", "vomiting", "diarrhea", "abdominal cramps", "mild fever"]),
    ("Food Poisoning",    ["vomiting", "diarrhea", "stomach pain", "nausea", "weakness"]),
    ("Appendicitis",      ["sharp abdominal pain", "nausea", "vomiting", "loss of appetite", "mild fever"]),
    ("Appendicitis",      ["abdominal pain", "vomiting", "nausea", "fever", "constipation"]),
    ("Acid Reflux",       ["heartburn", "chest pain", "regurgitation", "bloating", "sour taste"]),
    ("Acid Reflux",       ["heartburn", "sour taste", "bloating", "chest pain"]),
    ("Peptic Ulcer",      ["burning stomach pain", "bloating", "heartburn", "nausea", "loss of appetite"]),
    ("Peptic Ulcer",      ["stomach pain", "bloating", "nausea", "heartburn"]),
    ("Irritable Bowel Syndrome", ["abdominal pain", "bloating", "gas", "diarrhea", "constipation"]),
    ("Irritable Bowel Syndrome", ["bloating", "abdominal pain", "gas", "diarrhea"]),

    # ---------- Neurological ----------
    ("Migraine",          ["severe headache", "nausea", "light sensitivity", "vomiting", "blurred vision"]),
    ("Migraine",          ["headache", "nausea", "light sensitivity", "sound sensitivity"]),
    ("Tension Headache",  ["headache", "neck stiffness", "fatigue", "irritability"]),
    ("Tension Headache",  ["headache", "neck pain", "fatigue", "difficulty concentrating"]),
    ("Vertigo",           ["dizziness", "spinning sensation", "nausea", "balance problems", "vomiting"]),
    ("Vertigo",           ["dizziness", "balance problems", "nausea", "spinning sensation"]),

    # ---------- Cardiovascular ----------
    ("Hypertension",      ["headache", "dizziness", "blurred vision", "chest pain", "shortness of breath"]),
    ("Hypertension",      ["headache", "nosebleed", "dizziness", "fatigue"]),
    ("Angina",            ["chest pain", "shortness of breath", "fatigue", "dizziness", "nausea"]),
    ("Angina",            ["chest pain", "shortness of breath", "sweating", "nausea"]),
    ("Heart Attack",      ["severe chest pain", "shortness of breath", "sweating", "nausea", "left arm pain", "dizziness"]),
    ("Heart Attack",      ["chest pain", "left arm pain", "sweating", "shortness of breath", "nausea"]),

    # ---------- Metabolic / Endocrine ----------
    ("Diabetes",          ["excessive thirst", "frequent urination", "fatigue", "blurred vision", "weight loss", "slow healing wounds"]),
    ("Diabetes",          ["excessive thirst", "frequent urination", "fatigue", "blurred vision"]),
    ("Hypothyroidism",    ["fatigue", "weight gain", "cold intolerance", "constipation", "dry skin", "depression"]),
    ("Hypothyroidism",    ["fatigue", "weight gain", "cold intolerance", "dry skin"]),
    ("Hyperthyroidism",   ["weight loss", "rapid heartbeat", "heat intolerance", "anxiety", "sweating", "tremor"]),
    ("Hyperthyroidism",   ["weight loss", "rapid heartbeat", "anxiety", "sweating"]),

    # ---------- Musculoskeletal ----------
    ("Arthritis",         ["joint pain", "joint stiffness", "swelling", "reduced range of motion", "fatigue"]),
    ("Arthritis",         ["joint pain", "stiffness", "swelling", "warmth in joint"]),
    ("Gout",              ["sudden joint pain", "swelling", "redness", "warmth in joint", "tenderness"]),
    ("Gout",              ["joint pain", "redness", "swelling", "tenderness"]),
    ("Muscle Strain",     ["muscle pain", "swelling", "limited mobility", "bruising"]),
    ("Muscle Strain",     ["muscle pain", "swelling", "weakness"]),

    # ---------- Skin ----------
    ("Chickenpox",        ["itchy rash", "fever", "fatigue", "loss of appetite", "headache"]),
    ("Chickenpox",        ["rash", "fever", "itchy blisters", "fatigue"]),
    ("Measles",           ["high fever", "rash", "cough", "runny nose", "red eyes"]),
    ("Measles",           ["fever", "rash", "red eyes", "runny nose"]),
    ("Fungal Infection",  ["itchy skin", "redness", "rash", "peeling skin", "burning sensation"]),
    ("Fungal Infection",  ["itchy skin", "redness", "rash", "peeling skin"]),
    ("Eczema",            ["itchy skin", "dry skin", "red patches", "rash", "skin cracking"]),
    ("Eczema",            ["itchy skin", "dry skin", "red patches", "rash"]),
    ("Psoriasis",         ["scaly skin", "red patches", "itching", "dry skin", "thickened skin"]),
    ("Psoriasis",         ["scaly skin", "red patches", "itching", "dry skin"]),

    # ---------- Eye / ENT ----------
    ("Conjunctivitis",    ["red eyes", "itchy eyes", "watery eyes", "eye discharge", "gritty feeling"]),
    ("Conjunctivitis",    ["red eyes", "watery eyes", "eye discharge", "itchy eyes"]),
    ("Sinusitis",         ["facial pain", "nasal congestion", "headache", "runny nose", "reduced smell"]),
    ("Sinusitis",         ["nasal congestion", "facial pain", "headache", "cough"]),
    ("Tonsillitis",       ["sore throat", "difficulty swallowing", "fever", "swollen tonsils", "bad breath"]),
    ("Tonsillitis",       ["sore throat", "fever", "difficulty swallowing", "swollen tonsils"]),

    # ---------- Urinary ----------
    ("Urinary Tract Infection", ["burning urination", "frequent urination", "pelvic pain", "cloudy urine", "fever"]),
    ("Urinary Tract Infection", ["burning urination", "frequent urination", "pelvic pain", "cloudy urine"]),
    ("Kidney Stones",     ["severe back pain", "blood in urine", "nausea", "vomiting", "painful urination"]),
    ("Kidney Stones",     ["back pain", "blood in urine", "nausea", "painful urination"]),

    # ---------- Mental Health ----------
    ("Anxiety Disorder",  ["excessive worry", "restlessness", "fatigue", "difficulty concentrating", "rapid heartbeat", "sweating"]),
    ("Anxiety Disorder",  ["excessive worry", "restlessness", "rapid heartbeat", "difficulty concentrating"]),
    ("Depression",        ["persistent sadness", "loss of interest", "fatigue", "sleep problems", "difficulty concentrating", "appetite changes"]),
    ("Depression",        ["persistent sadness", "fatigue", "sleep problems", "loss of interest"]),
    ("Insomnia",          ["difficulty sleeping", "fatigue", "irritability", "difficulty concentrating"]),
    ("Insomnia",          ["difficulty sleeping", "daytime fatigue", "irritability"]),

    # ---------- Other ----------
    ("Anemia",            ["fatigue", "pale skin", "shortness of breath", "dizziness", "cold hands and feet"]),
    ("Anemia",            ["fatigue", "pale skin", "dizziness", "shortness of breath"]),
    ("Dengue",            ["high fever", "severe headache", "joint pain", "rash", "pain behind eyes", "nausea"]),
    ("Dengue",            ["high fever", "joint pain", "rash", "pain behind eyes"]),
    ("Malaria",           ["high fever", "chills", "sweating", "headache", "nausea", "muscle pain"]),
    ("Malaria",           ["high fever", "chills", "sweating", "headache", "muscle pain"]),
    ("Typhoid",           ["high fever", "weakness", "abdominal pain", "constipation", "headache", "loss of appetite"]),
    ("Typhoid",           ["high fever", "weakness", "abdominal pain", "headache"]),
    ("Hepatitis",         ["jaundice", "fatigue", "abdominal pain", "nausea", "loss of appetite", "dark urine"]),
    ("Hepatitis",         ["jaundice", "fatigue", "abdominal pain", "nausea"]),
]

# ==================================================================
# 2. DATA AUGMENTATION
#    Creates synthetic variants so each disease has enough
#    samples for reliable training + cross-validation.
# ==================================================================
def augment_data(raw, copies_per_case=8):
    """Generate variants by dropping 1-2 symptoms and adding mild noise."""
    augmented = []
    symptom_pool = sorted({s for _, syms in raw for s in syms})

    for disease, syms in raw:
        augmented.append((disease, list(syms)))          # keep original
        for _ in range(copies_per_case):
            variant = set(syms)
            # Drop 1-2 symptoms at random (patient doesn't report all)
            max_drop = min(2, len(variant) - 1)
            if max_drop > 0:
                drop_n = random.randint(0, max_drop)
                for _ in range(drop_n):
                    variant.discard(random.choice(list(variant)))
            # Occasionally add a mild noise symptom
            if random.random() < 0.25:
                variant.add(random.choice(symptom_pool))
            if variant:
                augmented.append((disease, sorted(variant)))
    return augmented

RAW_DATA = augment_data(RAW_DATA, copies_per_case=8)

# Unique symptom & disease lists
ALL_SYMPTOMS = sorted({s for _, syms in RAW_DATA for s in syms})
DISEASES     = sorted({d for d, _ in RAW_DATA})


def build_dataframe():
    """Convert RAW_DATA → one-hot encoded DataFrame."""
    rows = []
    for disease, syms in RAW_DATA:
        row = {s: 0 for s in ALL_SYMPTOMS}
        for s in syms:
            row[s] = 1
        row["prognosis"] = disease
        rows.append(row)
    return pd.DataFrame(rows)


# ==================================================================
# 3. TRAIN MODELS
#    Uses Stratified K-Fold cross-validation → no test_size error
# ==================================================================
@st.cache_resource(show_spinner=False)
def train_all_models():
    df = build_dataframe()
    X = df.drop("prognosis", axis=1)
    y = df["prognosis"]

    candidates = {
        "Naive Bayes"  : GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, max_depth=12
        ),
    }

    # Choose safe number of folds
    min_class_count = y.value_counts().min()
    n_splits = max(2, min(5, min_class_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results, trained = {}, {}
    for name, model in candidates.items():
        # Cross-validated predictions for honest metrics
        y_pred = cross_val_predict(model, X, y, cv=skf)
        results[name] = {
            "Accuracy" : accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "Recall"   : recall_score(y, y_pred, average="weighted", zero_division=0),
            "F1-Score" : f1_score(y, y_pred, average="weighted", zero_division=0),
        }
        # Final fit on ALL data for actual prediction
        model.fit(X, y)
        trained[name] = model

    best_name  = max(results, key=lambda k: results[k]["Accuracy"])
    best_model = trained[best_name]
    return best_model, best_name, pd.DataFrame(results).T, ALL_SYMPTOMS, df


# ==================================================================
# 4. STREAMLIT UI
# ==================================================================
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Based Disease Prediction System")
st.caption("Enter your symptoms below to receive a preliminary health assessment.")

st.warning(
    "⚠️ **Disclaimer:** This tool provides informational predictions only and "
    "is *not* a substitute for professional medical advice. Always consult a "
    "qualified doctor for accurate diagnosis and treatment."
)

with st.spinner("Training models on embedded dataset…"):
    model, best_name, metrics_df, symptoms, df = train_all_models()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ℹ️ System Info")
    st.write(f"**Best Model:** {best_name}")
    st.write(f"**Symptoms:** {len(symptoms)}")
    st.write(f"**Diseases:** {len(model.classes_)}")
    st.write(f"**Training cases:** {len(df)}")

    st.subheader("📊 Model Comparison")
    st.dataframe(
        metrics_df.style.format("{:.4f}"),
        use_container_width=True
    )

    with st.expander("🔬 View dataset sample"):
        st.dataframe(df.head(20), use_container_width=True)

# ---------------- Symptom Selection ----------------
pretty = [s.title() for s in symptoms]
display_to_raw = {s.title(): s for s in symptoms}

selected = st.multiselect(
    "🔍 Search and select your symptoms:",
    options=pretty,
    placeholder="Type to search (e.g., Headache, Fever, Cough)…",
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button(
        "🔎 Predict Disease",
        type="primary",
        disabled=not selected
    )
with col2:
    if st.button("🔄 Clear"):
        st.rerun()

# ---------------- Prediction ----------------
if predict_clicked:
    vector = np.zeros(len(symptoms))
    for s in selected:
        vector[symptoms.index(display_to_raw[s])] = 1

    probs   = model.predict_proba([vector])[0]
    classes = model.classes_
    top_idx = np.argsort(probs)[::-1][:3]
    top     = [(str(classes[i]), float(probs[i]) * 100) for i in top_idx]

    st.success("### 🧾 Prediction Result")

    if top[0][1] < 30:
        st.error(
            "Based on the symptoms provided, a confident prediction cannot be "
            "made. Please consider adding more symptoms or consulting a doctor."
        )

    medals = ["🥇", "🥈", "🥉"]
    for rank, (disease, conf) in enumerate(top, 1):
        st.markdown(
            f"{medals[rank-1]} **{rank}. {disease}** — "
            f"Confidence: **{conf:.2f}%**"
        )
        st.progress(min(conf / 100.0, 1.0))

    st.info(
        "**Note:** This is not a medical diagnosis. Please consult a qualified "
        "healthcare professional for accurate diagnosis and treatment."
    )

    with st.expander("🔍 Selected symptoms breakdown"):
        st.write(", ".join(sorted(selected)))

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "© 2026 Ankita (238330028) | Dr. Shakuntala Misra National "
    "Rehabilitation University, Lucknow"
)