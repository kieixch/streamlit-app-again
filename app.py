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
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import io

#App layout
st.set_page_config(page_title="DSS Klasifikasi (KNN) - Breast Cancer", layout="wide")
st.title("Decision Support System — Klasifikasi KNN")
st.markdown(
    """
    **Topik Default:** Prediksi diagnosis tumor payudara (Breast Cancer Wisconsin - Diagnostic).  
    **Metode:** K-Nearest Neighbors (KNN).  
    **Catatan:** Hanya untuk tujuan pendidikan / praktikum. Bukan untuk diagnosa medis.
    """
)

#Sidebar: Data Pilihan
st.sidebar.header("Pengaturan Data")
data_option = st.sidebar.selectbox("Pilih sumber dataset", ("Gunakan dataset default (Breast Cancer)", "Unggah file CSV"))

if data_option.startswith("Unggah"):
    uploaded_file = st.sidebar.file_uploader("Unggah file CSV (kolom fitur + kolom target bernama 'target')", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File CSV dimuat.")
    else:
        st.sidebar.info("Silakan unggah file CSV atau pilih dataset default.")
        df = None
else:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target 
    st.sidebar.write("Menggunakan dataset Breast Cancer (sklearn).")
    st.sidebar.write(f"Jumlah baris: {df.shape[0]}, fitur: {df.shape[1]-1}")

if df is None:
    st.stop()

st.subheader("Pratinjau data")
st.dataframe(df.head())

target = st.selectbox("Pilih kolom target:", df.columns)
X = df.drop(target, axis=1)
y = df[target]

#Sidebar: Preprocessing & Split
st.sidebar.header("Preprocessing & Split")
test_size = st.sidebar.slider("Persentase Test Set (%)", min_value=10, max_value=50, value=30, step=5, help="Gunakan 30% sesuai instruksi (default).")
random_state = st.sidebar.number_input("Random state (integer)", value=42, step=1)
stratify_option = st.sidebar.checkbox("Stratify split berdasarkan target (disarankan)", value=True)

if 'target' not in df.columns:
    st.error("Kolom target tidak ditemukan. Pastikan file CSV memiliki kolom bernama 'target' yang berisi label kelas (0/1 atau serupa).")
    st.stop()

X = df.drop(columns=['target'])
y = df['target']

st.write(f"Menekan split dengan test_size = {test_size}% (Train set = {100-test_size}%)")
if stratify_option:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state), stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))

st.write("Ukuran dataset:", "Train =", X_train.shape[0], " / Test =", X_test.shape[0])

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

#Sidebar: Model parameter
st.sidebar.header("Model KNN")
k_val = st.sidebar.slider("Pilih nilai k (n_neighbors)", min_value=1, max_value=21, value=5, step=1)
weights = st.sidebar.selectbox("weights", ("uniform", "distance"))
metric = st.sidebar.selectbox("metric", ("minkowski", "euclidean", "manhattan"))

#Train model
if st.sidebar.button("Latih & Evaluasi model"):
    knn = KNeighborsClassifier(n_neighbors=k_val, weights=weights, metric=metric)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)
    try:
        y_proba = knn.predict_proba(X_test_s)[:,1]
    except:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Hasil Evaluasi (Test Set)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1-score", f"{f1:.4f}")

    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            auc = roc_auc_score(y_test, y_proba)
            st.metric("ROC AUC", f"{auc:.4f}")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax2.plot([0,1],[0,1],'--', color='gray')
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend()
            st.pyplot(fig2)
        except Exception as e:
            st.warning("Tidak dapat menghitung ROC AUC: " + str(e))

    out_df = X_test.copy().reset_index(drop=True)
    out_df['actual'] = y_test.reset_index(drop=True)
    out_df['predicted'] = y_pred
    if y_proba is not None:
        out_df['proba_pos_class'] = y_proba

    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    st.download_button("Unduh hasil prediksi (CSV)", data=csv_buf.getvalue(), file_name="knn_predictions.csv", mime="text/csv")

    st.success("Selesai — model telah dilatih dan dievaluasi.")
else:
    st.info("Tekan tombol 'Latih & Evaluasi model' pada sidebar untuk memulai pelatihan dan evaluasi.")


#Footer
st.markdown("---")
st.markdown("**Instruksi singkat:** Gunakan slider `k` di sidebar, atur test size menjadi 30% (sesuai instruksi praktikum), lalu tekan 'Latih & Evaluasi model'.")
st.markdown("**Disclaimer:** Sistem ini dibuat untuk keperluan praktikum/pendidikan. Hasil prediksi bukan untuk diagnosa medis.")
