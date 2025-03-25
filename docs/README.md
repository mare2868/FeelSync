# FeelSync: Sistema de Recomendación Basado en Emociones

## Descripción
FeelSync es un sistema de recomendación híbrido que integra análisis emocional, procesamiento del lenguaje natural (NLP) y aprendizaje profundo para ofrecer recomendaciones personalizadas de contenido en función del estado emocional del usuario.

Este proyecto es el Trabajo de Fin de Máster desarrollado por Helen Carolina Roda Garcia y Lina Marcela Angel Cetina como parte de su formación en la Universidad Internacional de La Rioja (UNIR).

## Características Clave
- Clasificación de emociones con BERT.
- Recomendación híbrida con filtrado colaborativo y basado en contenido.
- Implementación de un prototipo interactivo en Streamlit.
- Integración de datasets de Netflix, MovieLens y Last.FM.

## Estructura del Proyecto
```
FeelSync/
├── data/               # Almacenamiento de datasets
│   ├── raw/            # Datasets originales sin procesar
│   ├── processed/      # Datasets limpios y transformados
│   ├── predictions/    # Datos con predicciones de modelos
│
├── notebooks/          # Jupyter notebooks para exploración y modelado
│   ├── 01-data-exploration.ipynb
│   ├── 02-feature-engineering.ipynb
│   ├── 03-model-training.ipynb
│   ├── 04-evaluation.ipynb
│
├── src/                # Código fuente modular
│   ├── data/           # Scripts de carga y procesamiento de datos
│   ├── features/       # Ingeniería de características
│   ├── models/         # Entrenamiento y evaluación de modelos
│   ├── visualizations/ # Generación de visualizaciones
│
├── tests/              # Pruebas unitarias
├── reports/            # Reportes y visualizaciones
├── docs/               # Documentación del proyecto
├── app/                # Implementación en Streamlit
│   ├── feelsync.py
│   ├── assets/
│
├── demo/               # Video demostrativo
│   ├── feelsync_demo_interfaz.mp4
│
├── requirements.txt    # Dependencias del proyecto
├── .gitignore          # Archivos a ignorar en Git
├── LICENSE             # Licencia del proyecto
└── architecture.md     # Explicación de la arquitectura del sistema
```

## Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/mare2868/FeelSync.git
cd FeelSync
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación en Streamlit
```bash
streamlit run app/feelsync.py
```

## Video de Demostración 🎬
Puedes ver una demo del funcionamiento de la interfaz en la carpeta [`demo/`](./demo).

📁 Archivo: `feelsync_demo_interfaz.mp4`  
🧭 Muestra cómo el usuario selecciona género y emoción para recibir recomendaciones personalizadas.

## Modelos Utilizados
- BERT: Para la clasificación de emociones.
- Red Neuronal Profunda: Para la generación de recomendaciones.
- Filtrado Colaborativo y Basado en Contenido: Para optimizar la personalización.

## Contribución
Si deseas mejorar FeelSync, puedes hacer un fork del repositorio y enviar un pull request con tus cambios.

## Licencia
Este proyecto está bajo la licencia MIT.

---

Hecho con ❤️ por Helen Carolina Roda Garcia y Lina Marcela Angel Cetina