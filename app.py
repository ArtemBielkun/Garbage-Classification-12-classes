import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps


st.set_page_config(page_title="EcoSort", page_icon="♻️", layout="wide")


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
    }
    
    .stApp { background-color: #e2f0e5; }
    
    .result-box {
        background-color: #ffffff;
        padding: 25px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        margin-bottom: 20px;
        border-left: 6px solid #2e7d32; 
    }
    .warning-box { border-left: 6px solid #d32f2f; }
    
    h1, h2, h3, h4 { color: #1b5e20; font-weight: 700; letter-spacing: -0.5px; }
    
    progress { width: 100%; height: 8px; border-radius: 0px; }
    progress::-webkit-progress-bar { background-color: #eeeeee; }
    progress::-webkit-progress-value { background-color: #2e7d32; }
    
    /* Картки для довідника контейнерів */
    .container-card {
        background-color: white; 
        padding: 20px; 
        margin-bottom: 15px; 
        border-radius: 8px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); 
        height: 140px;
    }
    </style>
""", unsafe_allow_html=True)


CLASS_NAMES = [
    "batteries", "biological", "brown-glass", "cardboard", "clothes", 
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

GARBAGE_INFO = {
    "paper": {"name": "ПАПІР", "color_hex": "#1976d2", "container": "СИНІЙ КОНТЕЙНЕР", "action": "Переконайтеся, що папір сухий і чистий, без залишків їжі чи жиру."},
    "cardboard": {"name": "КАРТОН", "color_hex": "#1976d2", "container": "СИНІЙ КОНТЕЙНЕР", "action": "Обов'язково розберіть та сплюсніть коробки."},
    "biological": {"name": "БІОВІДХОДИ", "color_hex": "#795548", "container": "КОРИЧНЕВИЙ КОНТЕЙНЕР", "action": "Викидати без пластикових пакетів."},
    "metal": {"name": "МЕТАЛ", "color_hex": "#f57f17", "container": "ЖОВТИЙ КОНТЕЙНЕР", "action": "Зімніть бляшанку. Переконайтеся, що всередині немає рідини."},
    "plastic": {"name": "ПЛАСТИК", "color_hex": "#f57f17", "container": "ЖОВТИЙ КОНТЕЙНЕР", "action": "Змийте залишки їжі, випустіть повітря та сплюсніть пляшку."},
    "green-glass": {"name": "ЗЕЛЕНЕ СКЛО", "color_hex": "#388e3c", "container": "ЗЕЛЕНИЙ КОНТЕЙНЕР", "action": "Обов'язково зніміть кришечки перед утилізацією."},
    "brown-glass": {"name": "КОРИЧНЕВЕ СКЛО", "color_hex": "#388e3c", "container": "ЗЕЛЕНИЙ КОНТЕЙНЕР", "action": "Обов'язково зніміть кришечки перед утилізацією."},
    "white-glass": {"name": "ПРОЗОРЕ СКЛО", "color_hex": "#388e3c", "container": "ЗЕЛЕНИЙ КОНТЕЙНЕР", "action": "Обов'язково зніміть кришечки перед утилізацією."},
    "clothes": {"name": "ОДЯГ", "color_hex": "#7b1fa2", "container": "БАК ДЛЯ ТЕКСТИЛЮ", "action": "Здавати у чистому та сухому вигляді."},
    "shoes": {"name": "ВЗУТТЯ", "color_hex": "#7b1fa2", "container": "БАК ДЛЯ ТЕКСТИЛЮ", "action": "Зв'яжіть пару взуття шнурками між собою."},
    "batteries": {"name": "БАТАРЕЙКИ", "color_hex": "#d32f2f", "container": "ПУНКТ ПРИЙОМУ (НЕБЕЗПЕЧНІ)", "action": "НЕБЕЗПЕЧНІ ВІДХОДИ. Категорично заборонено викидати в загальне сміття.", "danger": True},
    "trash": {"name": "ЗМІШАНЕ СМІТТЯ", "color_hex": "#424242", "container": "СІРИЙ АБО ЧОРНИЙ КОНТЕЙНЕР", "action": "Викидайте сюди все, що не підпадає під інші категорії."}
}


ALL_CONTAINERS = [
    {"name": "ЖОВТИЙ КОНТЕЙНЕР", "desc": "Пластик, метал, бляшанки, упаковка, тетрапак, поліетиленові пакети."},
    {"name": "СИНІЙ КОНТЕЙНЕР", "desc": "Папір, картон, газети, журнали, зошити, чисте паперове пакування."},
    {"name": "ЗЕЛЕНИЙ КОНТЕЙНЕР", "desc": "Будь-яке скло (пляшки від напоїв, банки від консервації)."},
    {"name": "КОРИЧНЕВИЙ КОНТЕЙНЕР", "desc": "Органічні залишки, очистки від овочів, фрукти, листя, залишки їжі."},
    {"name": "СІРИЙ / ЧОРНИЙ КОНТЕЙНЕР", "desc": "Залишкове змішане сміття, що не підлягає переробці (напр. брудні серветки)."},
    {"name": "БАК ДЛЯ ТЕКСТИЛЮ", "desc": "Одяг, взуття, постільна білизна, тканини (виключно у чистому вигляді)."},
    {"name": "ПУНКТ ПРИЙОМУ", "desc": "Батарейки, акумулятори, ртутні лампи, електроніка. НЕБЕЗПЕЧНО!"}
]


@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 12)
    try:
        model.load_state_dict(torch.load('model_weights_final.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


with st.sidebar:
    st.title("EcoSort")
    st.markdown("---")
    st.markdown("**МОДЕЛЬ:** ResNet50")
    st.markdown("**КЛАСІВ:** 12")
    st.markdown("**ТОЧНІСТЬ:** ~94%")


st.title("Метод автоматизованого контролю сортування відходів засобами глибокого навчання")
st.markdown("Завантажте зображення відходів для автоматичного аналізу та отримання інструкцій з утилізації.")

model = load_model()

if model is None:
    st.error("Системна помилка: Файл вагів 'model_weights_final.pth' не знайдено.")
else:
    uploaded_file = st.file_uploader("Оберіть файл (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        
        with col1:
            original_image = Image.open(uploaded_file).convert('RGB')
            square_image = ImageOps.fit(original_image, (600, 600), method=Image.Resampling.LANCZOS)
            st.image(square_image, caption='Завантажене зображення', use_column_width=True)
            
        
        with col2:
            with st.spinner("Обробка тензорів..."):
                img_tensor = transform(original_image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = F.softmax(outputs[0], dim=0)
                    confidence, predicted_idx = torch.max(probabilities, 0)
                    conf_percent = confidence.item() * 100
                    
                predicted_class_key = CLASS_NAMES[predicted_idx.item()]
                result = GARBAGE_INFO[predicted_class_key]
                
                box_class = "result-box warning-box" if result.get("danger") else "result-box"
                
                
                st.markdown(f"""
                <div class="{box_class}">
                    <p style="color: #666; font-size: 12px; font-weight: 600; margin-bottom: 0;">ІДЕНТИФІКОВАНИЙ КЛАС</p>
                    <h3 style="margin-top: 0; margin-bottom: 15px; color: #111;">{result['name']}</h3>
                    <p style="color: #666; font-size: 12px; margin-bottom: 5px;">Впевненість мережі: {conf_percent:.1f}%</p>
                    <progress value="{conf_percent}" max="100"></progress>
                </div>
                """, unsafe_allow_html=True)
                
                
                st.markdown(f"""
                <div class="result-box">
                    <p style="color: #666; font-size: 12px; font-weight: 600; margin-bottom: 0;">РЕКОМЕНДОВАНА УТИЛІЗАЦІЯ</p>
                    <h4 style="margin-top: 10px; margin-bottom: 15px; color: {result['color_hex']}; display: flex; align-items: center;">
                        <span style="display: inline-block; width: 14px; height: 14px; background-color: {result['color_hex']}; margin-right: 8px; border-radius: 2px;"></span>
                        {result['container']}
                    </h4>
                    <p style="font-size: 14px; margin-top: 0px; color: #333;"><b>Правило:</b> {result['action']}</p>
                </div>
                """, unsafe_allow_html=True)

        
        st.markdown("---")
        st.markdown("### ДОВІДНИК: ЩО В ЯКИЙ КОНТЕЙНЕР КИДАТИ")
        st.markdown("Звіряйтеся з цим списком, щоб дізнатися, які відходи приймає кожен контейнер.")
        
        
        cols = st.columns(3) 
        for i, cont in enumerate(ALL_CONTAINERS):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="container-card">
                    <strong style="color: #111; font-size: 14px;">{cont['name']}</strong>
                    <p style="color: #333; font-size: 14px; margin-top: 10px; line-height: 1.5;"><b>Приймає:</b> {cont['desc']}</p>
                </div>
                """, unsafe_allow_html=True)