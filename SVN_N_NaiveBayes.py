import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Generación de dataset no lineal (ideal para demostrar el Kernel RBF)
X_data, y_data = make_moons(n_samples=300, noise=0.22, random_state=42)

# 2. División en entrenamiento (70%) y prueba (30%) + Estandarización
# Nota: SVM con Kernel RBF es muy sensible a la escala de las variables
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.30, random_state=42, stratify=y_data
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entrenamiento y medición de tiempo: SVM (Kernel RBF)
t0 = time.perf_counter()
svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
tiempo_svm = (time.perf_counter() - t0) * 1000  # en milisegundos

# 4. Entrenamiento y medición de tiempo: Naive Bayes (Gaussiano)
t0 = time.perf_counter()
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
tiempo_nb = (time.perf_counter() - t0) * 1000  # en milisegundos

# 5. Evaluación de resultados
y_pred_svm = svm.predict(X_test_scaled)
y_pred_nb = nb.predict(X_test_scaled)

print("=== RESULTADOS PRÁCTICA 3.3: SVM Y NAIVE BAYES ===")
print(f"Precisión SVM (Kernel RBF): {accuracy_score(y_test, y_pred_svm)*100:.2f}% | Tiempo: {tiempo_svm:.3f} ms")
print(f"Precisión Naive Bayes:      {accuracy_score(y_test, y_pred_nb)*100:.2f}% | Tiempo: {tiempo_nb:.3f} ms\n")

print("--- Reporte detallado SVM ---")
print(classification_report(y_test, y_pred_svm))

print("--- Reporte detallado Naive Bayes ---")
print(classification_report(y_test, y_pred_nb))

# 6. Visualización de fronteras de decisión
def graficar_fronteras(modelos, titulos, X, y, archivo_salida="fronteras_svm_nb.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Crear malla de puntos para dibujar el fondo
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    for ax, modelo, titulo in zip(axes, modelos, titulos):
        Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.coolwarm)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=40)
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_xlabel("Característica 1 (escalada)")
        ax.set_ylabel("Característica 2 (escalada)")
        ax.grid(True, linestyle='--', alpha=0.4)
        
    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=150)
    print(f"Gráfica guardada exitosamente como '{archivo_salida}'")

graficar_fronteras(
    modelos=[svm, nb],
    titulos=[
        f"SVM - Kernel RBF (Acc: {accuracy_score(y_test, y_pred_svm)*100:.1f}%)",
        f"Gaussian Naive Bayes (Acc: {accuracy_score(y_test, y_pred_nb)*100:.1f}%)"
    ],
    X=X_test_scaled,
    y=y_test
)