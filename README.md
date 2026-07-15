# 👁️ JEPA-Foco v2: Biological Saccadic Attention Framework

Un clasificador de visión computacional ultra-ligero que emula el comportamiento visual biológico humano mediante la combinación de **Joint Embedding Predictive Architecture (JEPA)** y **Atención Sacádica**.

Este modelo procesa imágenes reales enfocando de manera selectiva su "fóvea virtual" en parches de alto valor de información, descartando hasta el 80% del cálculo de píxeles redundantes. Esto permite lograr inferencias precisas a velocidades de vértigo en hardware limitado (CPU / Termux / Linux Mint).

---

## 🔬 ¿Cómo Funciona la Magia? (Biología + IA)

A diferencia de las redes neuronales clásicas que analizan cada píxel de una imagen completa, **JEPA-Foco v2** emula el ojo humano:

1. **Periferia Rápida (Saliencia):** Un codificador JEPA ultra-ligero extrae un mapa de características globales a baja resolución y predice qué zonas de la imagen son "interesantes" en solo **~3.5 ms**.
2. **Sacadas Oculares:** El modelo selecciona los $N$ parches (coordenadas de cuadrícula $14 \times 14$) más relevantes y realiza saltos de atención rápidos (sacadas).
3. **Fóvea de Alta Definición:** Solo los parches seleccionados son procesados en alta resolución.
4. **Memoria de Trabajo:** Una celda recurrente (GRU) conecta y consolida la secuencia de "vistazos" para emitir un diagnóstico semántico final en **~7 ms**.

---

## 📁 Arquitectura Modular del Proyecto

El framework está diseñado bajo principios de ingeniería de software limpia, separando la definición de la red, el flujo de entrenamiento y la lógica de inferencia en tres módulos independientes:

```text
jepa_foco_v2/
├── engine.py        # 🧠 El Cerebro: Arquitectura neuronal del JEPA y la memoria de trabajo
├── entrenar.py      # 🏋️ El Entrenador: Carga datos, optimiza los pesos y guarda el modelo
├── predecir.py      # 🧪 El Laboratorio: Carga el modelo y evalúa imágenes nuevas con métricas de tiempo
└── data/            # 📁 Directorio de Datos (Organizado por el usuario)
    ├── train/       # Subcarpetas para entrenar (ej: train/gato/, train/perro/)
    └── test/        # Carpeta donde arrojas las imágenes nuevas a predecir

Explicación de los Componentes
​1. engine.py (La Red Neuronal)
​Define la arquitectura JEPAEngine. Implementa la capa de proyección lineal que emula el espacio latente del JEPA, el predictor de saliencia encargado de calcular el mapa de atención de la periferia ocre, y una memoria GRUCell que actúa como memoria de trabajo acumulando la información secuencial de cada sacada para la clasificación final.
​2. entrenar.py (Pipeline de Aprendizaje)
​Se encarga de escanear la carpeta data/train/, identificar de manera automática cuántas clases tiene el dataset mediante la estructura de carpetas físicas y entrenar la red por 50 épocas con retropropagación estándar utilizando el optimizador Adam. Al finalizar, exporta los pesos aprendidos de forma segura en modelo.pth.
​3. predecir.py (Herramienta de Auditoría e Inferencia)
​Es tu centro de pruebas de laboratorio. Carga el archivo modelo.pth, lee de forma dinámica la carpeta data/test/ y procesa cualquier imagen que coloques allí. Mide de forma independiente la latencia exacta de inferencia en milisegundos para verificar la eficiencia del modelo perimetral.
​🚀 Guía de Uso Rápido
​1. Preparación del Entorno
​Instala las dependencias estándar de visión y rendimiento en tu terminal:

pip install torch torchvision pillow tqdm

2. Organizar tus Imágenes
​Crea manualmente la estructura de carpetas y coloca tus fotos de la siguiente forma:
​Entrenamiento: data/train/gato/ (tus fotos de gatos) y data/train/perro/ (tus fotos de perros).
​Prueba (Examen Ciego): Coloca las fotos nuevas a identificar dentro de data/test/.
​3. Entrenar el Modelo
​Ejecuta el script de entrenamiento. Verás una barra de progreso interactiva por cada época:

python entrenar.py


Esto generará el archivo de pesos persistentes modelo.pth al finalizar.
​4. Predicción y Auditoría de Tiempos
​Para evaluar las imágenes de prueba y medir el rendimiento exacto del ojo virtual, ejecuta:

python predecir.py


El script imprimirá en la consola un informe detallado con este formato:

=== LABORATORIO DE PREDICCIÓN SACÁDICA ===

Imagen: test3.jpg
 -> Predicción: GATO (96.22%)
 -> Foco: ['(10,9)', '(1,5)', '(4,8)', '(10,11)']
 -> Latencia Inferencia: 10.17 ms

Imagen: dog3.jpg
 -> Predicción: PERRO (98.09%)
 -> Foco: ['(1,2)', '(1,3)', '(1,1)', '(1,11)']
 -> Latencia Inferencia: 11.23 ms


Métricas de Rendimiento en CPU (Single-Thread)
​En pruebas controladas de laboratorio local, el modelo reporta las siguientes latencias de procesamiento:
​Atención Periférica (Saliencia): ~3.35 ms
​Procesamiento de Fóvea + GRU: ~6.71 ms
​Inferencia Total: ~10.17 ms (Equivalente a procesar video en tiempo real a 90-100 FPS).

