import pandas as pd
from prophet import Prophet
import streamlit as st


def prevendo(df2, data1, flag=True):
    from prophet import Prophet
    import pandas as pd

    # Criando o modelo Prophet
    feriados_sp = pd.DataFrame({
        'holiday': 'feriados_sp',
        'ds': pd.to_datetime(['2024-01-25', '2024-02-25', '2024-03-25', '2024-04-25', '2024-05-25',
                              '2024-06-25', '2024-07-25', '2024-08-25', '2024-09-25', '2024-10-25',
                              '2024-11-25', '2024-12-25']),
        'lower_window': 0,
        'upper_window': 0,
    })

    m = Prophet(holidays=feriados_sp)

    # Adicionando feriados semanais (sábados e domingos)
    df2['ds'] = pd.to_datetime(df2['ds'], format='%d.%m.%Y')
    df2['is_weekend'] = (df2['ds'].dt.weekday >= 5).astype(int)
    m.add_regressor('is_weekend')

    m.fit(df2)  # Ajustando o modelo com o DataFrame original

    # Estendendo o período de previsão para incluir janeiro de 2024
    future_dates = pd.date_range(start='2024-01-01', periods=31, freq='D')
    future = pd.DataFrame({'ds': future_dates})
    future['is_weekend'] = (future['ds'].dt.weekday >= 5).astype(int)

    # Fazendo a previsão com base nas datas futuras
    forecast = m.predict(future)

    # Plotando os resultados para o período estendido
    fig_extended = m.plot(forecast)
    if flag:
        st.write(fig_extended)
    flag = False
    # Filtrando os resultados para a data desejada, excluindo feriados e finais de semana
    prediction1 = forecast[(forecast['ds'] == pd.to_datetime(data1)) &
                           (~forecast['ds'].isin(feriados_sp['ds'])) &
                           (~(forecast['ds'].dt.weekday >= 5))]

    if not prediction1.empty:
        yhat_period = prediction1['yhat'].values[0]
        return yhat_period
    else:
        return None





