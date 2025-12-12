# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import io

# ===========================
#   UI CONFIG
# ===========================
st.set_page_config(page_title="DSS KNN", layout="wide")
st.title("Decision Support System — Klasifikasi KNN")

st.markdown("""
**Metode:** K-Nearest Neighbors  
**Catatan:** Aplikasi untuk tugas/praktikum.  
""")

# ===========================
#   LOAD DATA
# ===========================
st.sidebar.header("Pengaturan Dataset")

data_option = st.sidebar.selectbox(
    "Pilih sumber dataset",
    ("Gunakan dataset default (Breast Cancer)", "Unggah file CSV")
)

df = None

if data_option.startswith("Unggah"):
    uploaded_file = st.sidebar.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV berhasil dimuat!")
        except Exception as e:
            st.sidebar.error(f"Gagal membaca CSV: {e}")
            df = None
else:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    st.sidebar.write("Dataset default digunakan.")
    st.sidebar.write(f"Jumlah baris: {df.shape[0]}")

if df is None:
    st.stop()

# ===========================
#   PREVIEW DATA
# ===========================
st.subheader("Pratinjau Data")
st.dataframe(df.head())

# ===========================
#   PILIH KOLOM TARGET
# ===========================
st.sidebar.header("Kolom Target")

all_columns = df.columns.tolist()
target_col = st.sidebar.selectbox("Pilih kolom target:", all_columns)

if df[target_col].nunique() < 2:
    st.error("Kolom target harus memiliki minimal 2 kelas.")
    st.stop()

# ===========================
#   SPLIT DATA
# ===========================
st.sidebar.header("Preprocessing & Split")

test_size = st.sidebar.slider("Test Size (%)", 10, 50, 30, 5)
random_state = st.sidebar.number_input("Random State", value=42)
stratify_option = st.sidebar.checkbox("Stratify", value=True)

X = df.drop(columns=[target_col])
y = df[target_col]

try:
    if stratify_option:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state
        )
except:
    st.warning("Stratify gagal, melakukan split biasa.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state
    )

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

st.write(f"Train: {X_train.shape[0]} baris, Test: {X_test.shape[0]} baris")

# ===========================
#   MODEL KNN
# ===========================
st.sidebar.header("Model KNN")
k_val = st.sidebar.slider("k (n_neighbors)", 1, 21, 5)
weights = st.sidebar.selectbox("weights", ("uniform", "distance"))
metric = st.sidebar.selectbox("metric", ("minkowski", "euclidean", "manhattan"))

train_btn = st.sidebar.button("Latih & Evaluasi Model")

# ===========================
#   TRAINING
# ===========================
if train_btn:

    try:
        knn = KNeighborsClassifier(n_neighbors=k_val, weights=weights, metric=metric)
        knn.fit(X_train_s, y_train)

        y_pred = knn.predict(X_test_s)

        try:
            y_proba = knn.predict_proba(X_test_s)[:, 1]
        except:
            y_proba = None

        # ===========================
        #   EVALUATION METRICS
        # ===========================
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.subheader("Hasil Evaluasi Model")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("Precision", f"{prec:.4f}")
        c3.metric("Recall", f"{rec:.4f}")
        c4.metric("F1-score", f"{f1:.4f}")

        # ===========================
        #   CONFUSION MATRIX
        # ===========================
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # ===========================
        #   VISUALISASI TAMBAHAN
        # ===========================
        st.markdown("## Visualisasi Tambahan")

        # --- ROC CURVE ---
        if y_proba is not None and df[target_col].nunique() == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax_roc.plot([0,1],[0,1],'--')
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

        # --- PRECISION-RECALL CURVE ---
        if y_proba is not None and df[target_col].nunique() == 2:
            precision, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            fig_pr, ax_pr = plt.subplots()
            ax_pr.plot(recall_vals, precision)
            ax_pr.set_title("Precision–Recall Curve")
            st.pyplot(fig_pr)

        # --- HISTOGRAM PROBABILITY ---
        if y_proba is not None:
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(y_proba, bins=20, color="skyblue", edgecolor="black")
            ax_hist.set_title("Distribusi Probabilitas Prediksi")
            st.pyplot(fig_hist)

        # ===========================
        #   PCA SCATTER (Fix label string)
        # ===========================
        st.markdown("### PCA 2D Scatter Plot (Actual vs Predicted)")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test_s)

        # factorize actual & predicted (convert class labels → angka)
        actual_codes, actual_labels = pd.factorize(y_test)
        pred_codes, pred_labels = pd.factorize(y_pred)

        fig_pca, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

        # Actual
        sc1 = ax1.scatter(X_pca[:,0], X_pca[:,1], c=actual_codes, cmap="tab10", edgecolor='k')
        ax1.set_title("Actual Labels")
        handles_a = [
            plt.Line2D([0],[0], marker='o', color='w',
                markerfacecolor=plt.cm.tab10(i/ max(1,len(actual_labels)-1)),
                markersize=7, markeredgecolor='k')
            for i in range(len(actual_labels))
        ]
        ax1.legend(handles_a, actual_labels, title="Actual")

        # Predicted
        sc2 = ax2.scatter(X_pca[:,0], X_pca[:,1], c=pred_codes, cmap="tab10", edgecolor='k')
        ax2.set_title("Predicted Labels")
        handles_p = [
            plt.Line2D([0],[0], marker='o', color='w',
                markerfacecolor=plt.cm.tab10(i/ max(1,len(pred_labels)-1)),
                markersize=7, markeredgecolor='k')
            for i in range(len(pred_labels))
        ]
        ax2.legend(handles_p, pred_labels, title="Predicted")

        st.pyplot(fig_pca)

        # ===========================
        #   BAR CHART METRIK
        # ===========================
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [acc, prec, rec, f1]
        })

        fig_bar, ax_bar = plt.subplots()
        sns.barplot(data=metrics_df, x="Metric", y="Value", ax=ax_bar)
        ax_bar.set_ylim(0,1)
        ax_bar.set_title("Perbandingan Metrik Evaluasi")
        st.pyplot(fig_bar)

        # ===========================
        #   DOWNLOAD CSV
        # ===========================
        out_df = X_test.copy()
        out_df["actual"] = y_test.values
        out_df["predicted"] = y_pred

        if y_proba is not None:
            out_df["proba_pos"] = y_proba

        buf = io.StringIO()
        out_df.to_csv(buf, index=False)
        st.download_button("Download Hasil Prediksi", buf.getvalue(), "prediksi.csv")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    st.info("Klik tombol **Latih & Evaluasi Model** untuk memulai.")

# FOOTER
st.markdown("---")
st.caption("Aplikasi ini dibuat untuk keperluan tugas & edukasi.")
