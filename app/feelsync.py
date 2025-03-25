import streamlit as st
import pandas as pd
import numpy as np
import random
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from PIL import Image

# Diccionario de emociones con emoticones
emotions_with_icons = {
    "🌟 Adventurous": "Adventurous",
    "😡 Angry": "Angry",
    "🤩 Excited": "Excited",
    "😂 Funny": "Funny",
    "😊 Happy": "Happy",
    "💖 Heartwarming": "Heartwarming",
    "🌱 Hopeful": "Hopeful",
    "✨ Inspiring": "Inspiring",
    "🤔 Nostalgic": "Nostalgic",
    "😍 Romantic": "Romantic",
    "😢 Sad": "Sad",
    "😱 Scared": "Scared",
    "🔎 Suspenseful": "Suspenseful"
}

# Mostrar imagen y título al inicio
try:
    image = Image.open("emociones.jpg")
except FileNotFoundError:
    image = None

col1, col2 = st.columns([1, 2])
with col1:
    if image:
        st.image(image, caption="Explora emociones", width=220)

with col2:
    st.markdown('<h1 style="color:#FF5733; ">FeelSync</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1C75BC; ">Sistema de Recomendación Personalizada</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:16px;  color:#333;">Descubre contenido adaptado a tus emociones</p>', unsafe_allow_html=True)


# Cargar dataset de Netflix
@st.cache_data
def cargar_datos_netflix():
    df2_netflix = pd.read_csv('netflix_predictions30.csv')
    df2_netflix['genres'] = df2_netflix['genres'].str.split(', ')

    # Reemplazar "Romantic TV Shows" por "Romance" para alinearlo con el diccionario
    df2_netflix['genres'] = df2_netflix['genres'].apply(
        lambda genres: [genre.replace("Romantic TV Shows", "Romance") for genre in genres]
    )

    # Codificar géneros
    genres_encoder_netflix = MultiLabelBinarizer()
    df2_netflix['genres_encoded'] = list(genres_encoder_netflix.fit_transform(df2_netflix['genres']))

    # Codificar emociones
    emotions_encoder_netflix = OneHotEncoder(sparse_output=False)
    df2_netflix['emotions_encoded'] = list(
        emotions_encoder_netflix.fit_transform(df2_netflix['predicted_emotions'].fillna('').values.reshape(-1, 1))
    )

    # Normalizar año de lanzamiento
    df2_netflix['release_year_norm'] = (df2_netflix['release_year'] - df2_netflix['release_year'].min()) / (
        df2_netflix['release_year'].max() - df2_netflix['release_year'].min()
    )

    # Extraer y normalizar duración
    df2_netflix['duration'] = df2_netflix['duration'].str.extract(r'(\d+)').astype(float)
    df2_netflix['duration_norm'] = (df2_netflix['duration'] - df2_netflix['duration'].min()) / (
        df2_netflix['duration'].max() - df2_netflix['duration'].min()
    )

    # Codificar títulos
    title_encoder_netflix = LabelEncoder()
    df2_netflix['title_encoded'] = title_encoder_netflix.fit_transform(df2_netflix['title'])

    return df2_netflix, genres_encoder_netflix, emotions_encoder_netflix, title_encoder_netflix


# Cargar dataset de Movies
@st.cache_data
def cargar_datos_movies():
    df_movies = pd.read_csv('movies_predictions20.csv')
    df_movies['genres'] = df_movies['genres'].str.split(', ').apply(lambda x: ', '.join(x))  # Convertir listas a cadenas
    df_movies['predicted_emotions'] = df_movies['predicted_emotions'].fillna('')

    # Codificar movieId y userId
    title_encoder_movies = LabelEncoder()
    df_movies['movieId'] = title_encoder_movies.fit_transform(df_movies['title'])

    user_encoder = LabelEncoder()
    df_movies['userId'] = user_encoder.fit_transform(df_movies['userId'])

    # One-hot encoding para géneros
    genres_encoder_movies = OneHotEncoder()
    genres_encoded = genres_encoder_movies.fit_transform(df_movies['genres'].values.reshape(-1, 1)).toarray()

    # One-hot encoding para emociones
    emotions_encoder_movies = OneHotEncoder()
    emotions_encoded = emotions_encoder_movies.fit_transform(df_movies['predicted_emotions'].values.reshape(-1, 1)).toarray()

    # Normalizar año
    df_movies['year_norm'] = (df_movies['year'] - df_movies['year'].min()) / (df_movies['year'].max() - df_movies['year'].min())

    return df_movies, title_encoder_movies, user_encoder, genres_encoder_movies, emotions_encoder_movies, genres_encoded, emotions_encoded

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    model_netflix = load_model("netflix_recommendation_model.h5", compile=False)
    model_movies = load_model("movie_recommendation_model.h5", compile=False)
    return model_netflix, model_movies

# Cargar datos y modelos
df2_netflix, genres_encoder_netflix, emotions_encoder_netflix, title_encoder_netflix = cargar_datos_netflix()
df_movies, title_encoder_movies, user_encoder_movies, genres_encoder_movies, emotions_encoder_movies, genres_encoded, emotions_encoded = cargar_datos_movies()
model_netflix, model_movies = cargar_modelos()

# Generar recomendaciones de Netflix
def generar_recomendaciones_netflix(df, genero, emocion, top_n=3):
    user_id = random.randint(1, 1000)

    # Filtrar títulos por género y emoción
    genre_mask = df['genres'].apply(
        lambda genres: any(genero.lower() in g.lower() for g in genres) if isinstance(genres, list) else False
    )
    emotion_mask = df['predicted_emotions'].apply(
        lambda emotions: any(emocion.lower() in e.lower() for e in str(emotions).split('; ')) if pd.notna(emotions) else False
    )
    filtered_titles = df[genre_mask & emotion_mask]

    if filtered_titles.empty:
        return "No se encontraron títulos de Netflix que coincidan con los criterios seleccionados."

    # Preprocesar entradas
    user_array = np.array([user_id] * len(filtered_titles))
    title_array = filtered_titles['title_encoded'].values
    genres_array = np.array(list(genres_encoder_netflix.transform(filtered_titles['genres'])))
    emotions_array = np.array(list(emotions_encoder_netflix.transform(filtered_titles['predicted_emotions'].fillna('').values.reshape(-1, 1))))
    release_year_array = filtered_titles['release_year'].astype(float).values
    duration_array = filtered_titles['duration'].astype(float).values

    # Entradas al modelo
    inputs = [user_array, title_array, genres_array, emotions_array, release_year_array, duration_array]

    # Predicción
    predicted_ratings = model_netflix.predict(inputs)

    # Normalización de calificaciones
    min_pred = predicted_ratings.min()
    max_pred = predicted_ratings.max()
    if min_pred == max_pred:
        predicted_ratings = np.full(predicted_ratings.shape, 3.5)
    else:
        predicted_ratings = 5 * (predicted_ratings - min_pred) / (max_pred - min_pred)
    predicted_ratings = np.clip(predicted_ratings, 0, 5).round(2)

    filtered_titles['predicted_rating'] = predicted_ratings

    # Ordenar y devolver recomendaciones
    recommended_titles = filtered_titles.sort_values(by='predicted_rating', ascending=False)
    return recommended_titles[['title', 'predicted_rating']].head(top_n)


# Generar recomendaciones de Movies
def generar_recomendaciones_movies(df, genero, emocion, top_n=3):
    # Filtrar títulos por género y emoción
    genre_mask = df['genres'].str.contains(genero, case=False)
    emotion_mask = df['predicted_emotions'].str.contains(emocion, case=False)
    filtered_movies = df[genre_mask & emotion_mask]

    if filtered_movies.empty:
        return "No se encontraron títulos de Movies que coincidan con los criterios seleccionados."

    # Eliminar duplicados para evitar repetir predicciones
    filtered_movies = filtered_movies.drop_duplicates(subset=['movieId'])

    # Seleccionar un user_id aleatorio
    user_id = random.choice(df['userId'].unique())
    user_array = np.array([user_id] * len(filtered_movies))
    movie_array = filtered_movies['movieId'].values
    genres_array = genres_encoder_movies.transform(filtered_movies['genres'].values.reshape(-1, 1)).toarray()
    emotions_array = emotions_encoder_movies.transform(filtered_movies['predicted_emotions'].values.reshape(-1, 1)).toarray()
    year_array = filtered_movies['year_norm'].values
    # Realizar predicciones
    inputs = [user_array, movie_array, genres_array, emotions_array, year_array]
    predicted_ratings = model_movies.predict(inputs)

    # Añadir calificaciones predichas
    filtered_movies['predicted_rating'] = np.clip(predicted_ratings, 0, 5)
    filtered_movies['predicted_rating'] = filtered_movies['predicted_rating'].round(2)

    # Ordenar y devolver recomendaciones
    recommended_titles = filtered_movies.sort_values(by='predicted_rating', ascending=False)
    return recommended_titles[['title', 'predicted_rating']].head(top_n)

# Interfaz

# Diccionario ajustado para mapear géneros con iconos a valores limpios
generos_dict = {
    "Action ⚔️": "Action",  # Común entre Movies y Netflix
    "Drama 🎭": "Drama",  # Común entre Movies y Netflix
    "Comedy 😂": "Comedy",  # Común entre Movies y Netflix
    "Romance ❤️": "Romance",  # "Romantic TV Shows" -> "Romance" en Netflix
    "Science Fiction 🛸": "Science Fiction",  # Solo en Movies
    "Suspense 🕵️‍♂️": "Suspense",  # Solo en Movies
    "Horror 👻": "Horror",  # "Horror Movies" -> "Horror" en Netflix
    "Family 👨‍👩‍👧‍👦": "Family",  # "Children & Family Movies" -> "Family" en Netflix
    "Fantasy 🪄": "Fantasy",  # Solo en Movies
    "Documentary 🎥": "Documentary",  # Común entre Movies y Netflix
    "Crime 🕶️": "Crime",  # "Crime TV Shows" -> "Crime" en Netflix
    "Thriller 🔪": "Thrillers",  # "Thrillers" en Netflix -> "Suspense" en Movies
    "Musical 🎶": "Musical",  # "Music & Musicals" -> "Musical" en Netflix
    "Independent 🎬": "Independent",  # Exclusivo de Netflix
    "International 🌏": "International",  # "International TV Shows" -> "International" en Netflix
    }



# Interfaz
# Opciones de plataforma
st.markdown("### 1. ¿Qué contenido prefieres?")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<h4 style="color: #FF5733;">🎥 Plataformas:</h4>', unsafe_allow_html=True)

with col2:
    opciones = st.multiselect(
        "Selecciona una o más plataformas:",
        options=["Netflix 🎥", "Movies 🎬"],
        default=["Netflix 🎥"],
        help="Elige las plataformas en las que deseas buscar contenido"
    )

# Opciones de género
st.markdown("### 2. ¿Qué género prefieres?")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<h4 style="color: #FF5733;">📚 Géneros:</h4>', unsafe_allow_html=True)

with col2:
    generos = list(generos_dict.keys())  # Usar géneros con iconos para la interfaz
    genero_seleccionado = st.selectbox(
        "Selecciona tu género favorito:",
        generos,
        help="Elige un género de tu preferencia"
    )
    genero = generos_dict[genero_seleccionado]  # Obtener el valor limpio para la lógica interna

# Opciones de emociones
st.markdown("### 3. ¿Cómo te sientes ahora?")
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<h4 style="color: #FF5733;">😊 Emoción:</h4>', unsafe_allow_html=True)

with col2:
    emocion_con_icono = st.selectbox(
        "Selecciona tu emoción actual:",
        list(emotions_with_icons.keys()),
        help="Elige la emoción que mejor describe tu estado"
    )
    emocion = emotions_with_icons[emocion_con_icono]  # Obtener el valor limpio

# Botón para generar recomendaciones
if st.button("Generar Recomendaciones"):
    if "Netflix 🎥" in opciones:  # Etiqueta ajustada para plataformas
        st.subheader("Recomendaciones de Netflix:")
        recomendaciones_netflix = generar_recomendaciones_netflix(df2_netflix, genero, emocion)
        if isinstance(recomendaciones_netflix, str):
            st.write(recomendaciones_netflix)
        else:
            for _, row in recomendaciones_netflix.iterrows():
                st.markdown(f"""
                🎬 **{row['title']}**  
                ⭐ Calificación: {row['predicted_rating']:.2f}  
                """)

    if "Movies 🎬" in opciones:  # Etiqueta ajustada para plataformas
        st.subheader("Recomendaciones de Movies:")
        recomendaciones_movies = generar_recomendaciones_movies(df_movies, genero, emocion)
        if isinstance(recomendaciones_movies, str):
            st.write(recomendaciones_movies)
        else:
            for _, row in recomendaciones_movies.iterrows():
                st.markdown(f"""
                🎬 **{row['title']}**  
                ⭐ Calificación: {row['predicted_rating']:.2f}  
                """)
