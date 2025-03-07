# Übersicht über die implementierten Gleichungen

Dieses Dokument fasst die wichtigsten Gleichungen zusammen, die zur Lösung der Mini‐Challenge (Gradient Descent mit neuronalen Netzen) implementiert wurden.

---

## 1. **Linearer Layer**

### Forward

**Gegeben**:  
- \(X\): Eingabe-Matrix \((N \times I)\)  
- \(W\): Gewichte \((I \times O)\)  
- \(B\): Bias-Vektor \((1 \times O)\)  
- \(N\): Anzahl der Samples (Batchgröße)  
- \(I\): Input-Dimension  
- \(O\): Output-Dimension  

\[
Z = X \cdot W + B
\]

Ergebnis: \(Z\) ist eine \((N \times O)\)-Matrix.

### Backward

**Gegeben**:  
- \(dOut\): Gradienten der Kostenfunktion bezüglich der Layer-Ausgabe \(Z\)

1. **Gradient bezüglich der Gewichte \(W\)**  
   \[
   \frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} \, dOut^\top \cdot X
   \]
2. **Gradient bezüglich des Bias \(B\)**  
   \[
   \frac{\partial \mathcal{L}}{\partial B} 
   = \frac{1}{N} \sum_{n=1}^{N} dOut_n
   \]
3. **Gradient bezüglich der Eingabe \(X\)**  
   \[
   \frac{\partial \mathcal{L}}{\partial X} 
   = dOut \cdot W
   \]

---

## 2. **Aktivierungsfunktionen**

### Sigmoid

**Forward**  
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

**Ableitung**  
\[
\sigma'(z) = \sigma(z)\,\bigl(1 - \sigma(z)\bigr)
\]

### ReLU

**Forward**  
\[
\mathrm{ReLU}(z) = \max(0, z)
\]

**Ableitung**  
\[
\mathrm{ReLU}'(z) = 
\begin{cases}
1 & \text{falls } z > 0,\\
0 & \text{sonst}
\end{cases}
\]

---

## 3. **Kostenfunktionen**

### Binary Cross‐Entropy (BCE)

Für die binäre Klassifikation (z. B. „Ziffer 7“ vs. „nicht 7“):

\[
L = -\frac{1}{N} \sum_{i=1}^{N} 
\Bigl[
  y_i \log(\hat{y}_i) \;+\; (1-y_i)\,\log\bigl(1-\hat{y}_i\bigr)
\Bigr]
\]

### Cross‐Entropy für Mehrklassen

Für die Mehrklassenklassifikation (z. B. 0–9 bei MNIST):

\[
L = -\frac{1}{N} \sum_{i=1}^{N} 
\sum_{c=1}^{C} y_{i,c} \log\bigl(\hat{y}_{i,c}\bigr)
\]

---

## 4. **Softmax**

Die Softmax‐Funktion wird häufig für die Mehrklassenklassifikation (z. B. MNIST 0–9) im letzten Layer verwendet:

\[
\mathrm{softmax}(z_i) 
= \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

mit \(z_i\) als i‐tes Element des Ausgabeverktors \(Z\).  

**Kombination Softmax + Cross‐Entropy**  
Vereinfacht die Ableitung, da der Gradient bezüglich \(Z\) dann \(\hat{y} - y\) ist, falls \(y\) und \(\hat{y}\) jeweils One‐Hot‐Vektoren sind.

---