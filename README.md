# Proyecto Final - Text Mining & Image Recognition

**Autor:** Fabio Humberto Suruy Juárez
**Carné:** 25001013
**Postgrado en Análisis y Predicción de Datos** - Universidad Galileo

Este repositorio contiene la solución a los dos ejercicios del proyecto final del curso.

## Problema 1 - Word Cloud

Carpeta: [`Problema 1 - Word Cloud/`](<Problema 1 - Word Cloud>)

- Notebook: [`Problema1_WordCloud.ipynb`](<Problema 1 - Word Cloud/Problema1_WordCloud.ipynb>)
- Dataset (1.6M tweets, formato Sentiment140): `tw_source.csv` (no incluido en el repo por su
  tamaño; descargar y colocar en esta carpeta antes de ejecutar).
- Resultados: carpeta [`outputs/`](<Problema 1 - Word Cloud/outputs>) con los 3 corpus por
  usuario, el resumen y los wordclouds generados.

**Metodologia:** "usuario popular" se calculo con regex de 2 formas: (A) usuario mas
**mencionado** (`@usuario` dentro del texto de otros tweets) y (B) autor mas **activo** (mas
tweets publicados). Se uso (A) como interpretacion principal porque es la que tiene sentido
para responder "¿por que citan a ese usuario?". Para el contexto de cada mencion se extrae una
**ventana real de palabras** alrededor de cada `@usuario` (no el tweet completo), se limpia
(stopwords + ruido de Twitter) y se compara **stemming vs. lematizacion** por separado (se usa
la version lematizada para el wordcloud final, por producir palabras reales y legibles).

Resumen de resultados (top 3 usuarios más mencionados vía `@usuario` en el texto de los tweets):

| Usuario | Menciones | Top 10 palabras de contexto (lematizado) |
|---|---|---|
| @mileycyrus | 4,580 | miley, love, hey, good, dont, come, vote, hope, luck, youre |
| @tommcfly | 3,904 | tom, say, hey, good, please, dont, love, dannymcfly, reply, haha |
| @ddlovato | 3,474 | demi, love, wish, hey, cant, hope, please, come, youre, selenagomez |

Como referencia, los 3 autores que **más tweets publicaron** (interpretación B) fueron
`@lost_dog` (549), `@webwoke` (345) y `@tweetpet` (310) — usuarios distintos a los citados
arriba, lo cual confirma que "más mencionado" y "más activo" son métricas diferentes.

## Problema 2 - Fruits and Vegetables Recognizer

Carpeta: [`Problema 2 - Fruits and Vegetables Recognizer/`](<Problema 2 - Fruits and Vegetables Recognizer>)

- Notebook: [`Problema2_CNN_FrutasVerduras.ipynb`](<Problema 2 - Fruits and Vegetables Recognizer/Problema2_CNN_FrutasVerduras.ipynb>)
- Dataset: [Fruits and Vegetables Dataset (Kaggle)](https://www.kaggle.com/) (no incluido en el
  repo por su tamaño).
- Frutas seleccionadas: Apple, Banana, Mango.
- Vegetales seleccionados: Carrot, Potato, Tomato.
- Se combinaron imágenes *Fresh* + *Rotten* de cada clase (~1,200 img/clase) para que el
  modelo aprenda el tipo de fruta/vegetal con variabilidad real de color/textura.

**Decisiones validadas con evidencia (no solo teoría):**
- **Color vs. escala de grises:** experimento controlado (Modelo A, 6 épocas) → RGB obtuvo
  80.98% de val_accuracy vs. 64.01% en escala de grises. Se confirma el uso de **color**.
- **Robustez ante variaciones fuertes** (rotación hasta 40°, zoom 0.4, brillo 0.4–1.6x) sobre
  el mejor modelo: el accuracy cayó solo **5.1 puntos porcentuales** (87.2% → 82.1%), señal de
  que el data augmentation de entrenamiento sí le dio tolerancia real a esas variaciones.

**Comparación de las 3 arquitecturas (test set, nunca visto en entrenamiento):**

| Modelo | Test Accuracy | Test Loss | Parámetros | Tiempo entrenamiento |
|---|---|---|---|---|
| **C - Profunda (GlobalAveragePooling)** | **87.2%** | 0.370 | 422,086 | 279s / 15 épocas |
| A - Baseline | 86.2% | 0.395 | 1,088,934 | 232s / 15 épocas |
| B - Regularizada (BatchNorm+Dropout) | 77.1% | 0.598 | 2,454,342 | 376s / 12 épocas (early stop) |

El Modelo C ganó con **menos parámetros** que A y B gracias al `GlobalAveragePooling2D`, que
actúa como regularizador estructural y da cierta invarianza a la posición del objeto en la
imagen. Resultados, matriz de confusión, ejemplos de predicciones y la curva de entrenamiento
están en [`outputs/`](<Problema 2 - Fruits and Vegetables Recognizer/outputs>).

## Auditoría y mejoras aplicadas

Ambos notebooks fueron auditados y corregidos tras una primera versión:

- **Problema 1:** la ventana de contexto alrededor de la mención no se estaba aplicando (se
  usaba el tweet completo); se reescribió con `TweetTokenizer` para tomar una ventana real de
  palabras. Además, encadenar `stem(lemmatize())` producía palabras rotas (p. ej. "please" →
  "plea"); ahora se muestran stemming y lematización por separado y el wordcloud final usa la
  versión lematizada. Se agregó también la comparación "usuario más mencionado" vs. "autor más
  activo" para dejar la interpretación documentada.
- **Problema 2:** se agregó medición de parámetros/tiempo por modelo, un experimento controlado
  color-vs-grises, una prueba cuantitativa de robustez ante variaciones fuertes, y una
  cuadrícula de predicciones de ejemplo (correctas/incorrectas).
- Se corrigió un bug de doble codificación UTF-8 (mojibake en tildes/¿/—) que ocurría al
  ejecutar los notebooks con `jupyter execute --inplace` en Windows sin `PYTHONUTF8=1`.

## Cómo ejecutar

Ambos notebooks fueron diseñados para correr en Google Colab (como pide el enunciado) o
localmente con Python 3.12 y las dependencias: `pandas`, `nltk`, `wordcloud`, `matplotlib`,
`scikit-learn`, `tensorflow`, `seaborn`, `jupyter`, `nbformat`.

```bash
pip install pandas nltk wordcloud matplotlib scikit-learn tensorflow seaborn jupyter nbformat
```
