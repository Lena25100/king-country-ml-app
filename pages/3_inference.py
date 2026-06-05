import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Инференс моделей", layout="wide")

st.title("Инференс и прогнозирование стоимости недвижимости")

@st.cache_resource
def load_all_models():
    models = {}
    models_dir = 'models'

    for i in [1, 2, 4, 5, 6]:
        filename = os.path.join(models_dir, f'model_ml{i}.pkl')
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    models[f'ML{i}'] = pickle.load(f)
            except Exception as e:
                models[f'ML{i}'] = f"Ошибка в файле {filename}: {e}"
        else:
            models[f'ML{i}'] = None

    cb_path = os.path.join(models_dir, 'model_ml3.cbm')
    if os.path.exists(cb_path):
        try:
            from catboost import CatBoostRegressor
            cb_model = CatBoostRegressor()
            cb_model.load_model(cb_path)
            models['ML3'] = cb_model
        except Exception as e:
            models['ML3'] = None
    else:
        models['ML3'] = None

    return models

models = load_all_models()

selected_model_name = st.selectbox(
    "Выберите модель для инференса:",
    ["ML1: Полиномиальная регрессия", 
     "ML2: Бустинг", 
     "ML3: CatBoost", 
     "ML4: Бэггинг", 
     "ML5: Стэкинг", 
     "ML6: Полносвязная нейросеть"]
)

model_key = selected_model_name.split(":")[0].strip()
current_model = models.get(model_key)

tab1, tab2 = st.tabs(["Ручной ввод параметров", "Загрузка CSV-файла"])

with tab1:
    st.write("### Задайте характеристики дома:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bedrooms = st.number_input("Количество спален", min_value=0, max_value=15, value=3, step=1)
        bathrooms = st.number_input("Количество ванных комнат", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
        floors = st.selectbox("Количество этажей", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=2)
        waterfront = st.radio("Выход к воде", [0, 1], format_func=lambda x: "Да (1)" if x == 1 else "Нет (0)")

    with col2:
        sqft_living = st.number_input("Жилая площадь (кв. футы)", min_value=300, max_value=15000, value=1800)
        st.caption(f"Это примерно {round(sqft_living * 0.0929, 1)} кв. метров")
        sqft_lot = st.number_input("Площадь участка (кв. футы)", min_value=500, max_value=1000000, value=5000)
        st.caption(f"Это примерно {round(sqft_lot * 0.0929, 1)} кв. метров")
        condition = st.slider("Техническое состояние (1-5)", 1, 5, 3)
        grade = st.slider("Класс дизайна и постройки (1-13)", 1, 13, 7)

    with col3:
        yr_built = st.number_input("Год постройки", min_value=1900, max_value=2026, value=1995, step=1)
        yr_renovated = st.number_input("Год ремонта (0, если не было)", min_value=0, max_value=2026, value=0, step=1)
        lat = st.number_input("Широта (lat)", value=47.56, format="%.4f")
        long = st.number_input("Долгота (long)", value=-122.21, format="%.4f")
        sale_year = st.number_input("Год продажи", min_value=2014, max_value=2025, value=2015)
        sale_month = st.number_input("Месяц продажи", min_value=1, max_value=12, value=6)

    sqft_above = sqft_living - 200 if sqft_living > 500 else sqft_living
    sqft_basement = 200 if sqft_living > 500 else 0
    sqft_living15 = sqft_living
    sqft_lot15 = sqft_lot
    view = st.slider("Оценка вида из окна", 0, 4, 0)
    
    input_data_19 = pd.DataFrame([{
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
        'sqft_lot': sqft_lot, 'floors': floors, 'waterfront': waterfront, 'view': view,
        'condition': condition, 'grade': grade, 'sqft_above': sqft_above,
        'sqft_basement': sqft_basement, 'yr_built': yr_built, 'yr_renovated': yr_renovated,
        'lat': lat, 'long': long, 'sqft_living15': sqft_living15, 'sqft_lot15': sqft_lot15,
        'year_sale': sale_year, 'month_sale': sale_month
    }])

    input_data_12 = pd.DataFrame([{
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
        'floors': floors, 'waterfront': waterfront, 'view': view, 'grade': grade,
        'sqft_above': sqft_above, 'sqft_basement': sqft_basement, 'yr_renovated': yr_renovated,
        'lat': lat, 'sqft_living15': sqft_living15
    }])
    
    st.markdown("---")
    if st.button("Рассчитать стоимость", type="primary"):
        if current_model is None:
            st.error(f"Выбранная модель ({model_key}) не загружена.")
        elif isinstance(current_model, str):
            st.error(current_model)
        else:
            try:
                if model_key == "ML6":
                    prediction = current_model.predict(input_data_12)
                else:
                    prediction = current_model.predict(input_data_19)
                
                predicted_price = float(prediction[0])

                if model_key in ['ML3', 'ML2', 'ML4', 'ML5', 'ML6'] and predicted_price < 50:
                    predicted_price = float(np.expm1(predicted_price))
                
                st.success("### Прогноз стоимости успешно сформирован!")
                st.metric(label="Стоимость дома", value=f"$ {predicted_price:,.2f}".replace(",", " "))
            except Exception as e:
                st.error(f"Ошибка в процессе инференса: {e}")

with tab2:
    st.write("### Предсказание из файла")
    uploaded_file = st.file_uploader("Загрузите тестовый файл в формате *.csv", type=["csv"])
    
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.dataframe(test_df.head(3))
        
        if st.button("Запустить расчет", type="primary"):
            if current_model is None or isinstance(current_model, str):
                st.error("Модель не доступна.")
            else:
                try:
                    if model_key == "ML6":
                        columns_to_use = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'grade', 'sqft_above', 'sqft_basement', 'yr_renovated', 'lat', 'sqft_living15']
                    else:
                        columns_to_use = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year_sale', 'month_sale']

                    missing_cols = [c for c in columns_to_use if c not in test_df.columns]

                    if missing_cols:
                        st.error(f"В файле отсутствуют необходимые колонки: {missing_cols}")
                    else:
                        X_batch = test_df[columns_to_use]
                        batch_preds = current_model.predict(X_batch)
                        
                        if model_key in ['ML3', 'ML2', 'ML4', 'ML5', 'ML6']:
                            batch_preds = np.expm1(batch_preds)
                            
                        result_df = test_df.copy()
                        result_df['Predicted_Price_USD'] = batch_preds
                        
                        st.success("Расчет окончен!")
                        st.dataframe(result_df[['sqft_living', 'bedrooms', 'Predicted_Price_USD']].head(10))
                        
                        csv_buffer = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать файл с прогнозами",
                            data=csv_buffer,
                            file_name="predictions_result.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Ошибка обработки файла: {e}")
                    