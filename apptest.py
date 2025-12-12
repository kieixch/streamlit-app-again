import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Star Classification – KNN Classifier With Custom Target Column")

st.markdown("""
Aplikasi ini menggunakan **star_classification.csv** sebagai dataset default.  
Anda juga dapat mengunggah dataset sendiri.  
Silakan pilih kolom mana yang menjadi **target klasifikasi**.
""")

# ----------------------------------------------------
# STEP 1: Load default dataset from CSV
# ----------------------------------------------------
def load_default_dataset():
    try:
        df = pd.read_csv("star_classification.csv")
        st.success("Default dataset 'star_classification.csv' berhasil dimuat.")
        return df
    except FileNotFoundError:
        st.error("❌ File 'star_classification.csv' tidak ditemukan! Pastikan file berada dalam folder yang sama dengan app.py.")
        return None

# Default load
df = load_default_dataset()

# ----------------------------------------------------
# STEP 2: Optional user-upload override
# ----------------------------------------------------
uploaded_file = st.file_uploader("Unggah dataset CSV Anda (opsional)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset berhasil dimuat dari file upload!")
    except:
        st.error("Tidak dapat membaca file CSV!")

# Stop if dataset is still None
if df is None:
    st.stop()

st.subheader("Preview Dataset")
st.dataframe(df)

# ----------------------------------------------------
# STEP 3: Select target column
# ----------------------------------------------------
columns = df.columns.tolist()
target_col = st.selectbox("Pilih kolom target (label)", options=columns)

if target_col is None:
    st.warning("Pilih kolom target untuk melanjutkan.")
    st.stop()

# ----------------------------------------------------
# STEP 4: Split data
# ----------------------------------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

try:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
except:
    st.error("Kolom target tidak bisa diproses. Pastikan isinya tidak kosong.")
    st.stop()

test_size = st.slider("Test Size (%)", 10, 50, 30)
k_min = st.slider("Nilai K Minimum", 1, 10, 3)
k_max = st.slider("Nilai K Maksimum", 5, 20, 11)

if k_min >= k_max:
    st.warning("Nilai K Minimum harus lebih kecil dari K Maksimum.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=test_size/100, random_state=42, stratify=y_encoded
)

# ----------------------------------------------------
# STEP 5: Preprocessing
# ----------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# STEP 6: Run KNN for each k
# ----------------------------------------------------
results = []

for k in range(k_min, k_max + 1):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results.append([k, acc, prec, rec, f1])

results_df = pd.DataFrame(results, columns=["k", "Accuracy", "Precision", "Recall", "F1-Score"])

st.subheader("📊 Hasil Evaluasi Untuk Setiap K")
st.dataframe(results_df)

# ----------------------------------------------------
# STEP 7: Confusion Matrix for best K
# ----------------------------------------------------
best_k = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["k"]

st.success(f"K terbaik berdasarkan akurasi: **k = {int(best_k)}**")

best_model = KNeighborsClassifier(n_neighbors=int(best_k))
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best)

st.subheader(f"Confusion Matrix Untuk K = {int(best_k)}")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

st.markdown("### Kesimpulan")
st.markdown(f"""
Model terbaik terdapat pada **k = {int(best_k)}**  
dengan metrik akurasi maksimum = **{results_df['Accuracy'].max():.4f}**
""")
