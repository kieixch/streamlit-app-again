import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from PIL import Image, ImageFilter

st.set_page_config(page_title="Star/Galaxy Classification", layout="wide")
st.title("⭐ Star / Galaxy / QSO Classification (Tabular + Image Detector)")
st.markdown("""
**Metode:** K-Nearest Neighbors  
**Catatan:** 
obj_ID = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS
alpha = Right Ascension angle (at J2000 epoch)
delta = Declination angle (at J2000 epoch)
u = Ultraviolet filter in the photometric system
g = Green filter in the photometric system
r = Red filter in the photometric system
i = Near Infrared filter in the photometric system
z = Infrared filter in the photometric system
run_ID = Run Number used to identify the specific scan
rereun_ID = Rerun Number to specify how the image was processed
cam_col = Camera column to identify the scanline within the run
field_ID = Field number to identify each field
spec_obj_ID = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
class = object class (galaxy, star or quasar object)
redshift = redshift value based on the increase in wavelength
plate = plate ID, identifies each plate in SDSS
MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
fiber_ID = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation.  
""")

# ===================================================================
# SYNTHETIC IMAGE GENERATOR (NO CV2)
# ===================================================================

def gaussian_blob(size, cx, cy, sigma, amp=1.0):
    x = np.arange(0, size); y = np.arange(0, size)
    xx, yy = np.meshgrid(x, y)
    return amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2*sigma*sigma))

def make_star(size=128):
    canvas = np.zeros((size, size))
    for _ in range(2):
        cx = np.random.uniform(size*0.4, size*0.6)
        cy = np.random.uniform(size*0.4, size*0.6)
        canvas += gaussian_blob(size, cx, cy, sigma=np.random.uniform(1,3),
                                amp=np.random.uniform(0.7,1.2))
    canvas += np.random.normal(0, 0.015, (size, size))
    canvas = (canvas - canvas.min()) / (canvas.max()+1e-9)
    return Image.fromarray((canvas*255).astype(np.uint8)).convert("L")

def make_galaxy(size=128):
    canvas = np.zeros((size, size))
    cx = cy = size/2
    canvas += gaussian_blob(size, cx, cy, sigma=10, amp=1.0)
    for t in np.linspace(0, 4*np.pi, 80):
        r = t * 3
        x = cx + r*np.cos(t)
        y = cy + r*np.sin(t)
        canvas += gaussian_blob(size, x, y, sigma=2, amp=0.20)
    canvas += np.random.normal(0, 0.02, (size, size))
    canvas = (canvas - canvas.min()) / (canvas.max()+1e-9)
    img = Image.fromarray((canvas*255).astype(np.uint8)).convert("L")
    return img.filter(ImageFilter.GaussianBlur(radius=1.4))

def make_qso(size=128):
    canvas = np.zeros((size, size))
    cx = cy = size/2
    canvas += gaussian_blob(size, cx, cy, sigma=2, amp=1.3)
    for _ in range(4):
        canvas += gaussian_blob(size,
                                np.random.uniform(0,size),
                                np.random.uniform(0,size),
                                sigma=np.random.uniform(0.5,1.8),
                                amp=0.25)
    canvas += np.random.normal(0, 0.03, (size, size))
    canvas = (canvas - canvas.min()) / (canvas.max()+1e-9)
    return Image.fromarray((canvas*255).astype(np.uint8)).convert("L")

# ===================================================================
# FEATURE EXTRACTION V2 (NO CV2 — WORKS ON STREAMLIT)
# ===================================================================

def extract_features_v2(img):
    img = img.resize((128,128)).convert("L")
    arr = np.array(img) / 255.0

    features = [
        arr.mean(), arr.std(), arr.max(), arr.min(),
        np.percentile(arr, 90), np.percentile(arr, 50)
    ]

    # Manual Laplacian edges
    lap = (
        np.abs(arr[:-2,1:-1] - arr[2:,1:-1]) +
        np.abs(arr[1:-1,:-2] - arr[1:-1,2:])
    )
    features.append(lap.mean())
    features.append(lap.std())

    # Bright core ratio
    features.append((arr > 0.75).sum() / arr.size)

    # Radial features
    h,w = arr.shape
    cx, cy = h//2, w//2
    yy,xx = np.mgrid[:h,:w]
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    features.append(r.mean())
    features.append(r.std())

    # Gradient features
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx**2 + gy**2)
    features.append(grad.mean())
    features.append(grad.std())
    features.append(np.percentile(grad, 90))

    # Symmetry score
    features.append(np.mean(np.abs(arr - np.flip(arr, axis=1))))

    # Fourier Transform features
    fft = np.abs(np.fft.fft2(arr))
    features.append(fft.mean())
    features.append(np.percentile(fft, 95))

    return np.array(features)

# ===================================================================
# TRAINING THE IMAGE MODEL
# ===================================================================

def generate_class_dataset_v2(n=600):
    X=[]; y=[]
    for _ in range(n):
        X.append(extract_features_v2(make_star()));   y.append("STAR")
        X.append(extract_features_v2(make_galaxy())); y.append("GALAXY")
        X.append(extract_features_v2(make_qso()));    y.append("QSO")
    return np.array(X), np.array(y)

@st.cache_resource
def train_image_model_v2():
    X, y = generate_class_dataset_v2()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=250)
    model.fit(Xs, y_enc)

    return model, scaler, le

img_model, img_scaler, img_labeler = train_image_model_v2()

# ===================================================================
# LOAD DEFAULT TABULAR DATASET
# ===================================================================

try:
    df = pd.read_csv("star_classification.csv")
    st.success("Dataset default dimuat: star_classification.csv")
except:
    st.error("❌ File 'star_classification.csv' tidak ditemukan!")
    st.stop()

st.subheader("📌 Dataset Default")
st.dataframe(df)

# ===================================================================
# TABULAR CLASSIFICATION
# ===================================================================

target_col = st.selectbox("Pilih kolom target", df.columns)

X = df.drop(columns=[target_col])
y = df[target_col]

le = LabelEncoder()
y_enc = le.fit_transform(y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

scaler = StandardScaler()
Xtrain_s = scaler.fit_transform(Xtrain)
Xtest_s = scaler.transform(Xtest)

st.subheader("🔎 Evaluasi KNN Multi-K")

k_min = st.slider("K Minimum", 1, 5, 3)
k_max = st.slider("K Maksimum", 5, 20, 11)

results = []

for k in range(k_min, k_max+1):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(Xtrain_s, Ytrain)
    pred = model.predict(Xtest_s)

    results.append([
        k,
        accuracy_score(Ytest, pred),
        precision_score(Ytest, pred, average="weighted", zero_division=0),
        recall_score(Ytest, pred, average="weighted", zero_division=0),
        f1_score(Ytest, pred, average="weighted", zero_division=0)
    ])

res_df = pd.DataFrame(results, columns=["k","Accuracy","Precision","Recall","F1"])
st.dataframe(res_df)

best_k = res_df.sort_values("Accuracy", ascending=False).iloc[0]["k"]
st.success(f"K terbaik: **{int(best_k)}**")

best_model = KNeighborsClassifier(n_neighbors=int(best_k))
best_model.fit(Xtrain_s, Ytrain)
best_pred = best_model.predict(Xtest_s)
cm = confusion_matrix(Ytest, best_pred)

# ===================================================================
# VISUALIZATION
# ===================================================================

st.header("📊 Visualisasi Analisis")

# 1. Line chart
st.subheader("📈 Line Chart – Metrics vs K")
fig1, ax1 = plt.subplots()
ax1.plot(res_df["k"], res_df["Accuracy"], label="Accuracy")
ax1.plot(res_df["k"], res_df["Precision"], label="Precision")
ax1.plot(res_df["k"], res_df["Recall"], label="Recall")
ax1.plot(res_df["k"], res_df["F1"], label="F1")
ax1.legend()
st.pyplot(fig1)

# 2. Bar chart
st.subheader("📊 Bar Chart Evaluasi")
fig2, ax2 = plt.subplots()
res_df.plot(x="k", y=["Accuracy","Precision","Recall","F1"], kind="bar", ax=ax2)
st.pyplot(fig2)

# 3. Confusion Matrix
st.subheader("🔵 Confusion Matrix (K terbaik)")
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax3)
st.pyplot(fig3)

# 4. Correlation Heatmap
st.subheader("🔥 Heatmap Korelasi Numerik")
numeric_df = df.select_dtypes(include=["int64","float64"])
fig4, ax4 = plt.subplots()
if numeric_df.shape[1] >= 2:
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax4)
else:
    st.warning("Tidak cukup fitur numerik untuk heatmap.")
st.pyplot(fig4)

# ===================================================================
# IMAGE DETECTOR
# ===================================================================

st.header("🖼️ Image Detector (STAR / GALAXY / QSO) – Tanpa OpenCV")

uploaded = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L").resize((128,128))
    st.image(img, width=250, caption="Gambar Uploaded")

    feat = extract_features_v2(img).reshape(1,-1)
    feat_scaled = img_scaler.transform(feat)

    pred = img_model.predict(feat_scaled)[0]
    label = img_labeler.inverse_transform([pred])[0]

    st.success(f"🔍 Prediksi: **{label}**")
