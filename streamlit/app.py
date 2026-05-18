import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Classificador de Deepfakes",
    page_icon="🖼️",
    layout="centered",
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../modelo-EfficientNet/saved_model/deepfake_model_EfficientNet.keras")


model = load_model()

CLASSES = ["FAKE", "REAL"]
INPUT_SIZE = (256, 256)


def preprocessar(imagem: Image.Image) -> np.ndarray:
    """Pré-processa a imagem antes de enviá-la ao modelo."""
    img = imagem.convert("RGB")
    img = img.resize(INPUT_SIZE)

    # Converte para float32
    arr = np.array(img, dtype=np.float32)

    arr = arr / 255.0

    # Adiciona a dimensão do batch (1, 256, 256, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr


def prever(imagem: Image.Image):
    """Executa a inferência e retorna o resultado formatado."""
    entrada = preprocessar(imagem)

    # O modelo retorna um único valor (probabilidade de ser classe 1: REAL)
    prob_real = float(model.predict(entrada)[0][0])
    prob_fake = 1.0 - prob_real

    # Se a probabilidade de ser REAL for > 50%, a classe predita é REAL
    if prob_real > 0.5:
        classe_pred = "REAL"
        confianca = prob_real
    else:
        classe_pred = "FAKE"
        confianca = prob_fake

    # Retorna as probabilidades na mesma ordem de CLASSES ["FAKE", "REAL"]
    return classe_pred, confianca, [prob_fake, prob_real]


# interface
st.title("🖼️ Classificador de Deepfakes")
st.write(
    "Faça upload de uma imagem **ou** tire uma foto pela câmera "
    "para obter a predição do modelo."
)

st.divider()
fonte = st.radio(
    "Origem da imagem",
    options=["Upload de arquivo", "Câmera"],
    horizontal=True,
)

imagem_pil = None
if fonte == "Upload de arquivo":
    arquivo = st.file_uploader("Selecione uma imagem", type=[
                               "jpg", "jpeg", "png", "webp"])
    if arquivo is not None:
        imagem_pil = Image.open(arquivo)
else:
    foto = st.camera_input("Tire uma foto")
    if foto is not None:
        imagem_pil = Image.open(foto)

if imagem_pil is not None:
    st.divider()
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Imagem recebida")
        st.image(imagem_pil, use_container_width=True)

    with col_result:
        st.subheader("Resultado")
        with st.spinner("Processando..."):
            classe, confianca, probs = prever(imagem_pil)

        if classe == "REAL":
            st.success(f"**Classe predita:** {classe}")
        else:
            st.error(f"**Classe predita:** {classe}")

        st.metric("Confiança", f"{confianca * 100:.1f}%")

        st.divider()
        st.write("**Probabilidades por classe:**")
        for nome, prob in zip(CLASSES, probs):
            st.progress(
                float(prob),
                text=f"{nome}: {prob * 100:.1f}%",
            )
else:
    st.info("Aguardando imagem...")

st.divider()
st.caption("Modelo de predição de deepfakes")
