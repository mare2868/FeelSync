# Arquitectura del Sistema de Recomendación FeelSync

## 1. Introducción
FeelSync es un sistema de recomendación híbrido basado en emociones, utilizando **BERT** para la clasificación de emociones y un modelo de **red neuronal profunda** para generar recomendaciones. 

## 2. Componentes Claves

### 🔹 **1. Ingreso de Datos**
   - Se utilizan datasets de **Netflix, MovieLens y Last.FM**.
   - Se dividen en **datos sin procesar (raw), transformados (processed) y predicciones generadas (predictions).**

### 🔹 **2. Clasificación de Emociones con BERT**
   - Se usa **BERT fine-tuned** para clasificar emociones en base a reseñas y comentarios de usuarios.
   - Se generan embeddings para representar emociones numéricamente.

### 🔹 **3. Sistema de Recomendación Híbrido**
   - **Filtrado basado en contenido:** Recomendaciones según características de las películas/música.
   - **Filtrado colaborativo:** Se analiza el comportamiento de usuarios similares.
   - **Predicción de calificaciones:** Se usa una **red neuronal profunda** para predecir qué tan adecuado es un contenido para un usuario.

## 3. Flujo del Sistema

```
Usuario 🧑‍💻 → Estado Emocional 🎭 → Clasificación con BERT 🧠 → Generación de Embeddings 🔢 → Sistema de Recomendación 🎬🎶 → Contenido Personalizado 📌
```

### **Modelo de Red Neuronal**
🔸 **Capa de Entrada:**
   - Embeddings de usuario y contenido.
   - Representación numérica de emociones.
   - Atributos normalizados (año de lanzamiento, duración, etc.).

🔸 **Capas Ocultas:**
   - Red neuronal densa con 3 capas (**128, 64 y 32 neuronas**).
   - Activación **SELU** y regularización con **Dropout**.

🔸 **Capa de Salida:**
   - Predicción de calificaciones del usuario para distintos contenidos.

## 4. Implementación en Streamlit
- Se desarrolla una **interfaz interactiva** que permite al usuario seleccionar su estado emocional y recibir recomendaciones personalizadas.
- Se muestran **gráficos** y explicaciones sobre las recomendaciones generadas.

## 5. Futuras Mejoras
✅ **Optimizar la recomendación de música** incorporando embeddings de audio.
✅ **Desplegar el sistema en una plataforma en la nube (GCP o AWS).**
✅ **Incluir IA generativa para explicar las recomendaciones.**

---
Con esta arquitectura, FeelSync ofrece una experiencia de recomendación **innovadora y personalizada, combinando inteligencia artificial y emociones.** 🚀