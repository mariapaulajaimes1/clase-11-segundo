import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas


def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result


st.set_page_config(page_title='🎨 Reconocimiento de Dígitos Escrito a Mano 🧠', layout='wide')

# Título y subtítulos llamativos
st.title("🖌️ Reconocimiento de Dígitos con IA 🎉")
st.markdown("#### Dibuja un dígito en el panel y haz clic en **Predecir** para ver el resultado! 🚀")


with st.sidebar:
    st.title("ℹ️ Acerca de esta Aplicación")
    st.markdown("""
    🎨 Esta aplicación evalúa la habilidad de una Red Neuronal Artificial para reconocer dígitos escritos a mano. 
    💡 Basada en el trabajo de Vinay Uniyal.
    """)


st.write("### 🎈 Zona de Dibujo Interactiva")
stroke_width = st.slider("🖍️ Selecciona el Ancho de Línea:", 1, 50, 20)
drawing_mode = "freedraw"
stroke_color = "#FFFFFF"  
bg_color = "#000000"

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.3)", 
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=500, 
    width=500, 
    drawing_mode=drawing_mode,
    key="canvas",
)


if st.button("🔍 Predecir"):
    if canvas_result.image_data is not None:
       
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype("uint8"), "RGBA")
        input_image = input_image.convert("L") 
        result = predictDigit(input_image)
        st.header(f"🔢 El dígito predicho es: **{result}**")
    else:
        st.warning("⚠️ Por favor, dibuja un dígito en el lienzo antes de predecir.")


st.markdown("""
    <style>
    .stButton button {
        background-color: #FF4500;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        width: 100%;
        border: none;
    }
    .stButton button:hover {
        background-color: #FF6347;
    }
    .reportview-container {
        background-color: #FFF5EE;
    }
    .sidebar .sidebar-content {
        background-color: #FFDAB9;
    }
    </style>
    """, unsafe_allow_html=True)

