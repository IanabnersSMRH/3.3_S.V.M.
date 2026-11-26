# 3.3 Support Vector Machines (SVM) and Naive Bayes
# Práctica 3.3: Máquinas de Vectores de Soporte (SVM) y Naive Bayes

Script en Python con `scikit-learn` que compara el rendimiento y las fronteras de decisión de **SVM (Kernel RBF)** y **Gaussian Naive Bayes** sobre un conjunto de datos no lineal (`make_moons`).

## Resultados rápidos
- **SVM (Kernel RBF):** ~87.78% de precisión en prueba (~4.6–5.1 ms de entrenamiento), logrando una frontera curva adaptada a los datos.
- **Gaussian Naive Bayes:** ~83.33% de precisión en prueba (~1.1–1.2 ms de entrenamiento), con menor tiempo de cómputo y una frontera más simple.

## Archivos
- `SVN_N_NaiveBayes.py`: Entrenamiento, evaluación y generación de la gráfica.
- `fronteras_svm_nb.png`: Comparativa visual de las fronteras de decisión.
