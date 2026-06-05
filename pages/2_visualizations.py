import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Визуализация", layout="wide")

st.title("Визуализация зависимостей в данных")

data_path = os.path.join('data', 'king_county_processed.csv')

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    
    st.write("### 1. Распределение целевой переменной (Цена)")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['price'], bins=50, kde=True, ax=ax, color='blue')
    st.pyplot(fig)
    
    st.write("### 2. Зависимость цены от жилой площади")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x='sqft_living', y='price', alpha=0.5, ax=ax, color='green')
    st.pyplot(fig)
    
    st.write("### 3. Распределение количества спален")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x='bedrooms', ax=ax)
    st.pyplot(fig)
    
    st.write("### 4. Матрица корреляции признаков")
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.error("Загрузите датасет для отображения графиков.")
    