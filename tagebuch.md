# Lerntagebuch

## Woche 1  
Diese Woche haben wir uns intensiv mit der Aufgabenstellung auseinandergesetzt und offene Begriffe gesammelt.  
Danach haben wir begonnen, uns mit den theoretischen Grundlagen des **Gradient Descent** und **neuronaler Netze** vertraut zu machen – unter anderem mithilfe der empfohlenen Videos von *3blue1brown* und dem bereitgestellten Python-Minimal-Template.  
Parallel dazu haben wir erste **Rechnungen zur Funktionsweise eines Linearlayers** (Forward- und Backward-Pass) durchgeführt und dokumentiert, um eine fundierte Basis für die spätere Programmierung zu schaffen.

---

## Woche 2  
Wir haben die erste Aufgabe abgeschlossen: das **Laden und Erkunden des MNIST-Datensatzes** mit `torchvision`.  
Mit `matplotlib` konnten wir einzelne Bilder visualisieren und die **Verteilung der Ziffernklassen** analysieren.  
Im Anschluss konvertierten wir die Daten in `NumPy`-Arrays, um sie in den folgenden Aufgaben weiterverarbeiten zu können.  
Außerdem finalisierten wir unsere **händische Beispielrechnung** für einen einfachen Linearlayer mit zwei Inputs und zwei Neuronen – als Vorbereitung für die Implementierung und spätere Tests.

---

## Woche 3  
Diese Woche stand im Zeichen der praktischen Umsetzung von **Aufgabe 2**.  
Wir haben eine **Linearlayer-Klasse in Python mit NumPy** programmiert – inklusive Methoden für `forward`, `backward` und `update`.  
Besonderes Augenmerk legten wir auf **Unit-Tests**, die auf unserer zuvor erstellten händischen Berechnung basierten.  
So konnten wir sicherstellen, dass unsere Implementierung exakt den erwarteten Ergebnissen entsprach.  
Kleinere Rechenfehler bei den Gradienten konnten wir mithilfe gezielter Tests schnell beheben.

---

## Woche 4  
Wir haben ein einfaches **neuronales Netzwerk mit einem Hidden Layer** erstellt, um eine bestimmte Ziffer (wir wählten die **7**) zu erkennen.  
Dabei bestand der Output nur aus einem Neuron, das 1 (für „7“) bzw. 0 (für andere Ziffern) zurückgeben sollte.  
Wir integrierten unsere Linearlayer-Klasse in ein Trainings-Setup, definierten eine geeignete **Kostenfunktion (MSE)** und eine Evaluationsfunktion (Accuracy), und begannen mit dem Training.  
Durch Variation der **Lernraten (0.01–1)** und **Hidden Layer-Größen (4, 8, 16)** konnten wir interessante Effekte auf den Lernverlauf beobachten – z. B. Instabilität bei zu hoher Lernrate.  
Wir hielten unsere Erkenntnisse sorgfältig fest und diskutierten mögliche Ursachen und Verbesserungen.

---

## Woche 5  
In der letzten Woche erweiterten wir unser Netzwerk auf **drei Hidden Layer** mit gleicher Neuronenzahl und **10 Output-Knoten**, um alle Ziffern des MNIST-Datensatzes zu klassifizieren.  
Zusätzlich implementierten wir **Mini-Batch-Training**, um effizienter mit den Daten zu arbeiten.  
Wir führten Trainingsläufe mit verschiedenen **Lernraten (0.001–0.1)** und **Layergrößen (16, 32, 64)** durch und verglichen die Resultate.  
Ein zu großer Hidden Layer führte teils zu Overfitting, während zu kleine Netzwerke die Trainingsdaten nicht ausreichend modellieren konnten.  
Wir entschieden uns für eine mittlere Größe bei moderater Lernrate als besten Kompromiss.  
Am Ende dokumentierten wir alle Ergebnisse, reflektierten unseren Lernprozess und verfassten die finale Abgabe mit den nötigen mathematischen Definitionen und Überlegungen.

---
