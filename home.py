import streamlit as st
import os

st.set_page_config(page_title="РГР: Машинное обучение", layout="wide")

st.title("Расчетно-графическая работа")
st.subheader("Дисциплина: «Машинное обучение и большие данные»")
st.write("**Тема:** Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")

st.divider()

col1, col2 = st.columns([1, 1]) 
with col1:
    photo_path = os.path.join('images', 'photo.jpg')
    if os.path.exists(photo_path):
        st.image(photo_path, caption='Рыговская Елена, Разработчик') 
    else:
        st.info("Здесь будет ваше фото")
with col2:
    st.markdown("### Студент:")
    st.write("**ФИО:** Рыговская Елена Александровна")
    st.write("**Группа:** ФИТ-242")
    