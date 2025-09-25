import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import os

# 1. Load data
df = pd.read_csv('heart.csv')

# Tampilkan 5 data teratas dalam bentuk tabel
print("=" * 80)
print("5 DATA TERATAS DARI DATASET PENYAKIT JANTUNG")
print("=" * 80)
print(df.head().to_string(index=False))
print("=" * 80)
print()

X = df.drop('target', axis=1)
y = df['target']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Validasi sebelum melanjutkan
while True:
    lanjut = input("Lanjutkan Progress? (y/n): ").lower().strip()
    if lanjut in ['y', 'ya', 'yes']:
        break
    elif lanjut in ['n', 'no', 'tidak']:
        print("Program berhenti.")
        exit()
    else:
        print("Input tidak valid! Masukkan 'y' untuk lanjut atau 'n' untuk berhenti.")

# 4. Menu input model
while True:
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Pilih Model Machine Learning:")
    print("1. Logistic Regression")
    print("2. Decision Tree")
    print("3. SVM (Linear Kernel)")
    choice = input("Masukkan pilihan (1/2/3): ")
    if choice == '1':
        model = LogisticRegression(max_iter=1000, random_state=42)
        model_name = "LOGISTIC REGRESSION"
        break
    elif choice == '2':
        model = DecisionTreeClassifier(random_state=42)
        model_name = "DECISION TREE"
        break
    elif choice == '3':
        model = SVC(kernel='linear', probability=True, random_state=42)
        model_name = "SVM (LINEAR KERNEL)"
        break
    else:
        print("Input tidak valid! Silakan masukkan 1, 2, atau 3.")
        input("Tekan Enter untuk mengulangi...")

# 5. Latih & evaluasi model
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean()

# 6. Pie chart data
counts = df['target'].value_counts().sort_index()
labels_pie = ['TIDAK TERKENA PENYAKIT JANTUNG', 'TERKENA PENYAKIT JANTUNG']
colors_pie = ['#66b3ff', '#ff9999']
total = counts.sum()
persentase_tidak = counts[0] / total * 100
persentase_terkena = counts[1] / total * 100

# 7. Plot visualisasi
fig = plt.figure(figsize=(14, 8))
fig.suptitle(f'AKURASI {model_name}', fontsize=16, fontweight='bold', y=0.95)

# --- Bar Chart ---
ax1 = plt.axes([0.14, 0.6, 0.6, 0.3])  # [left, bottom, width, height] dalam persentase
bars = ax1.barh(
    ['CV MEAN ACCURACY', 'TEST ACCURACY'],
    [cv_mean, acc],
    color=['#00bfff', '#ff6f69'],
    height=0.8,
    edgecolor='black'
)

# Tambahkan persentase di ujung bar
for bar in bars:
    width = bar.get_width()
    offset = 0.01 if width < 0.98 else -0.08  # jika 100%, geser ke kiri
    ax1.text(width + offset, bar.get_y() + bar.get_height()/2,
             f"{width*100:.1f}%", va='center', fontsize=11)

ax1.set_xlim(0, 1.05)
ax1.set_xlabel('Akurasi')
ax1.set_title('')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='y', labelsize=12)

# --- Pie Chart ---
ax2 = plt.subplot2grid((3, 4), (0, 3), rowspan=2)
wedges, texts, autotexts = ax2.pie(
    counts,
    autopct='%1.0f%%',
    startangle=90,
    colors=colors_pie,
    textprops={'fontsize': 12}
)
ax2.axis('equal')
ax2.set_title("")

# Legend Pie Chart
legend_patch = [
    mpatches.Patch(color=colors_pie[0], label='TIDAK TERKENA PENYAKIT JANTUNG'),
    mpatches.Patch(color=colors_pie[1], label='TERKENA PENYAKIT JANTUNG')
]
ax2.legend(handles=legend_patch, loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=False)

# --- NOTE Section ---
note_text = f"""
NOTE :
1. Test Accuracy (Akurasi Test Set)
   Akurasi model saat diuji dengan data yang belum pernah dilihat sebelumnya (data test).

2. CV Mean Accuracy (Cross-Validation Mean Accuracy)
   Rata-rata akurasi dari beberapa pengujian silang (cross-validation) selama pelatihan model.

JADI :
Perbandingan orang yang terkena penyakit jantung dengan orang yang tidak terjangkit
penyakit jantung adalah {persentase_terkena:.0f}% : {persentase_tidak:.0f}%
"""

ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
ax3.axis('off')
ax3.text(0, 1, note_text, va='top', fontsize=11, family='monospace')

# Layout akhir
plt.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)
plt.show()
