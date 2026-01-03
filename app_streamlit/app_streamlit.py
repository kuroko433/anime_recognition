import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import pandas as pd
# ==================== CONFIGURACIÓN DE LA APP ====================
st.set_page_config(
    page_title="Anime Character Recognizer",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🎨 Reconocedor de Personajes Anime")
st.markdown("**Arrastra una imagen o haz clic para subirla** · 17 personajes de diferentes animes · Accuracy ~96%")

# ==================== LISTA DE PERSONAJES ====================
# ¡¡¡ CAMBIA ESTA LISTA POR TUS 16 PERSONAJES REALES !!!
@st.cache_data
def load_class_names():
    url_clases = "https://raw.githubusercontent.com/kuroko433/anime_recognition/main/classes/clases_2.csv"
    df = pd.read_csv(url_clases)
    
    # Convertir a lista simple de strings
    # ADAPTA ESTO SEGÚN TU CSV:
    class_names = df.iloc[:, 0].tolist()  # Si no tiene encabezado: primera columna
    
    # O si tiene encabezado y la columna se llama 'personaje', 'name', etc.:
    # class_names = df['personaje'].tolist()
    
    return class_names

class_names = load_class_names()
# ==================== CARGA DEL MODELO DESDE GITHUB ====================
@st.cache_resource(show_spinner="Cargando el modelo desde GitHub...")
def load_model():
    # URL de tus pesos (cámbiala cuando lo subas)
    model_url = "https://raw.githubusercontent.com/kuroko433/anime_recognition/main/models/anime_classifier_17chars.pth"
    
    # Descargar pesos
    state_dict = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")
    
    # Recrear EXACTAMENTE como en create_model con extra_hidden_layers=0
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    
    # Obtener in_features
    in_features = model.classifier[1].in_features  # 1280
    
    # Reemplazar classifier por uno SIN Dropout, solo Linear directo (como hiciste tú)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features, 17)  # Directo: 1280 → 17 clases
    )
    
    # Cargar tus pesos (ahora las claves coinciden perfectamente)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

model = load_model()

# ==================== TRANSFORMACIONES DE IMAGEN ====================
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Tamaño óptimo para EfficientNetV2-S
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==================== INTERFAZ DRAG & DROP ====================
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Mostrar imagen subida
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True, caption="Imagen subida")

    # Clasificación
    with st.spinner("Analizando la imagen..."):
        img_tensor = transform(image).unsqueeze(0)  # [1, 3, 384, 384]

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Predicción principal
            confidence, predicted_idx = torch.max(probabilities, 0)
            pred_class = class_names[predicted_idx.item()]
            conf_percent = confidence.item() * 100

            # Top 3
            top3_probs, top3_idx = torch.topk(probabilities, 3)

    # ==================== RESULTADOS ====================
    st.success(f"**¡Predicción: {pred_class}!**")
    st.metric(label="Confianza", value=f"{conf_percent:.1f}%")

    st.markdown("### Top 3 predicciones")
    cols = st.columns(3)
    for i, col in enumerate(cols):
        idx = top3_idx[i].item()
        prob = top3_probs[i].item() * 100
        char = class_names[idx]
        with col:
            if i == 0:
                st.markdown(f"🥇 **{char}**")
            elif i == 1:
                st.markdown(f"🥈 **{char}**")
            else:
                st.markdown(f"🥉 **{char}**")
            st.progress(prob / 100)
            st.caption(f"{prob:.1f}%")

else:
    st.info("👆 Arrastra una imagen aquí o haz clic para seleccionar")
    st.markdown("""
    ### Ejemplos que funcionan bien:
    - Capturas de anime
    - Fanarts de buena calidad
    - Ilustraciones oficiales
    - Cosplays frontales
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Modelo entrenado con PyTorch + EfficientNetV2-S · Full fine-tuning · Dataset curado manualmente · 2025")