import matplotlib
from matplotlib import font_manager

# Clear and rebuild the font cache
font_manager._rebuild()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Загружаем данные
@st.cache_data
def load_data():
    file_url = "http://95.169.204.69:8092/csv/0xc18360217d8f7ab5e7c516566761ea12ce7f9d72/0xc18360217d8f7ab5e7c516566761ea12ce7f9d72.csv"
    df = pd.read_csv(file_url)
    df['value'] = df['value'].astype(float)
    df['ts'] = pd.to_datetime(df['ts'])
    return df

df = load_data()

# Функция загрузки данных
@st.cache_data
def load_data_features():
    file_id = "1ozjVKd5Vcca2yXcCRC0TYVQKT2s320ne"
    file_url_2 = f"https://drive.google.com/uc?id={file_id}"
    df = pd.read_csv(file_url_2)
    return df

df_features = load_data_features()

# Рассчитываем баланс и активность каждого кошелька
wallet_balances = {}
wallet_tx_counts = {}
wallet_tx_volumes = {}

for _, row in df.iterrows():
    from_wallet = row['from_address']
    to_wallet = row['to_address']
    value = row['value']

    wallet_balances[from_wallet] = wallet_balances.get(from_wallet, 0) - value
    wallet_balances[to_wallet] = wallet_balances.get(to_wallet, 0) + value
    
    wallet_tx_counts[from_wallet] = wallet_tx_counts.get(from_wallet, 0) + 1
    wallet_tx_counts[to_wallet] = wallet_tx_counts.get(to_wallet, 0) + 1
    
    wallet_tx_volumes[from_wallet] = wallet_tx_volumes.get(from_wallet, 0) + value
    wallet_tx_volumes[to_wallet] = wallet_tx_volumes.get(to_wallet, 0) + value

wallets = list(wallet_balances.keys())
balances = np.array([wallet_balances[w] for w in wallets])
tx_counts = np.array([wallet_tx_counts.get(w, 0) for w in wallets])
tx_volumes = np.array([wallet_tx_volumes.get(w, 0) for w in wallets])

# Интерфейс Streamlit
st.title("📊 Анализ блокчейн-кошельков")
st.subheader("1. Проверка баланса кошельков")

# Выпадающий список с кошельками
wallet_address = st.selectbox("Выберите кошелек:", wallets)

if wallet_address:
    wallet_df = df[(df['from_address'] == wallet_address) | (df['to_address'] == wallet_address)]
    
    if not wallet_df.empty:
        # Отображаем текущий баланс кошелька
        latest_balance = wallet_balances.get(wallet_address, 0)
        latest_price = wallet_df.iloc[-1]['price'] if 'price' in wallet_df.columns else 0
        latest_value = latest_balance * latest_price
        
        st.metric("🔍 Текущий баланс монет", f"{latest_balance:.2f}")
        st.metric("🔍 Последняя цена монеты", f"{latest_price:.2f}")
        st.metric("🔍 Текущий баланс в валюте", f"{latest_value:.2f}")
st.markdown("---")
# Кластеризация с учетом количества и объема транзакций
st.subheader("2. Кластеризация кошельков (учет объема и количества транзакций)")

# Добавление пояснения и изображения
st.markdown("""
### Определение китов, маркетмейкеров и толпы

**1.Киты:**
- Крупные суммы транзакций
- Низкая частота
- Значительный чистый поток

**2.Маркетмейкеры:**
- Высокая частота
- Небольшие суммы транзакций
- Сбалансированные потоки

**3.Другие пользователи:**
- Промежуточные показатели

**Модель кластеризации:**
Используется алгоритм K-Means с нормализацией данных. Группы выявляются на основе количества и объема транзакций.
""")
st.image("kmeans.png", caption="Механизм кластеризации", use_container_width=True)
st.markdown("---")
st.subheader("3. Визуализации кластеров")

# Визуализация кластеров
colors = {"whale": "orange", "market_maker": "coral", "other": "gray"}
fig = px.scatter(df_features, x="total_volume", y="total_count", color="role",
                title="Кластеризация адресов", color_discrete_map=colors)
st.plotly_chart(fig)

# Столбчатый график с процентами
role_counts = df_features["role"].value_counts(normalize=True) * 100
fig_bar = px.bar(role_counts, x=role_counts.index, y=role_counts.values,
                title="Распределение адресов (в %)", labels={"x": "Группа", "y": "Процент адресов (%)"},
                color=role_counts.index, color_discrete_map=colors)
fig_bar.update_layout(yaxis=dict(showgrid=True, gridcolor="lightgray", gridwidth=0.5))
st.plotly_chart(fig_bar)

# Столбчатый график с логарифмической шкалой
role_counts_log = df_features["role"].value_counts()
fig_bar_log = px.bar(role_counts_log, x=role_counts_log.index, y=role_counts_log.values,
                    title="Распределение адресов (логарифмическая шкала)", labels={"x": "Группа", "y": "Количество адресов"},
                    color=role_counts_log.index, color_discrete_map=colors)
fig_bar_log.update_layout(yaxis_type="log", yaxis=dict(showgrid=True, gridcolor="lightgray", gridwidth=0.5))
st.plotly_chart(fig_bar_log)

st.write(df_features["role"].value_counts())
