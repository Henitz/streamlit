# import plotly.express as px
import warnings

# !pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
# !pip install statsmodels
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import acf, pacf
import streamlit as st

# Ignorar os FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def modelo(df1):
    import pandas as pd
    from prophet import Prophet

    # Suponha que você tenha um DataFrame 'df' com as colunas 'ds' (datas) e 'y' (valores)
    # ...

    # m = Prophet()

    # Adicionando feriados nacionais brasileiros
    feriados_sp = pd.DataFrame({
        'holiday': 'feriados_sp',
        'ds': pd.to_datetime(['2024-01-25', '2024-02-25', '2024-03-25', '2024-04-25', '2024-05-25',
                              '2024-06-25', '2024-07-25', '2024-08-25', '2024-09-25', '2024-10-25',
                              '2024-11-25', '2024-12-25']),
        'lower_window': 0,
        'upper_window': 0,
    })

    # Criando o modelo Prophet
    m = Prophet(holidays=feriados_sp)

    # Converter 'ds' para o formato de data, se necessário
    df1['ds'] = pd.to_datetime(df1['ds'])

    # Adicionando feriados semanais (sábados e domingos)
    # Cria campo is_weekend com 0 e 1 - significando fim de semana
    df1['is_weekend'] = (df1['ds'].dt.weekday >= 5).astype(int)
    m.add_regressor('is_weekend')

    m.fit(df1)

    # Criando o dataframe para previsão futura
    future = m.make_future_dataframe(periods=365)
    future['is_weekend'] = (future['ds'].dt.weekday >= 5).astype(int)
    forecast = m.predict(future)

    # Conversão para arrays para uso em plotagem
    fcst_t = np.array(forecast['ds'].dt.to_pydatetime())
    history_ds = np.array(m.history['ds'].dt.to_pydatetime())

    fig, ax = plt.subplots()
    ax.plot(history_ds, m.history['y'], 'k.', label='Histórico')
    ax.plot(fcst_t, forecast['yhat'], label='Previsão')

    # Personalizando o gráfico
    ax.set_xlabel('Data')
    ax.set_ylabel('Valores')
    ax.legend()
    st.pyplot(fig)
    # Mostrando

    # Calculando métricas de desempenho
    # df_cv = performance_metrics(df)
    # print(df_cv.head())
    # este procedimento está dando erro

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Verificar se os DataFrames têm o mesmo número de amostras
    # df e forecast são diferentes pois forecast foi levado em conta os feridos e fins de semana
    # if df1.shape[0] == forecast.shape[0]:
    # Calcular métricas
    #   mae = mean_absolute_error(df1['y'], forecast['yhat'])
    #  mse = mean_squared_error(df1['y'], forecast['yhat'])
    #  rmse = np.sqrt(mse)

    # sf.write(f"MAE: {mae}")
    # sf.write(f"MSE: {mse}")
    # sf.write(f"RMSE: {rmse}")
    # else:
    #   df.write("Os DataFrames não têm o mesmo número de amostras. Verifique os dados.")
    # Calculando as previsões do modelo para os dados de teste
    # Calculando novamente forecast para todos os dados, vai ser prossivel calcular as métricas
    forecast = m.predict(df1)

    # Calculando o MAE entre as previsões e os valores reais
    mae = mean_absolute_error(df1['y'], forecast['yhat'])
    mae_rounded = round(mae, 2)

    st.write(f'MAE: {mae_rounded}')
    mse = mean_squared_error(df1['y'], forecast['yhat'])
    mse_rounded = round(mse, 2)
    rmse = mean_squared_error(df1['y'], forecast['yhat'], squared=False)
    rmse_rounded = round(rmse)
    st.write(f'MSE: {mse_rounded}')
    st.write(f'RMSE: {rmse_rounded}')


def prevendo(df2, data1):
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

    # Filtrando os resultados para a data desejada, excluindo feriados e finais de semana
    prediction1 = forecast[(forecast['ds'] == pd.to_datetime(data1)) &
                           (~forecast['ds'].isin(feriados_sp['ds'])) &
                           (~(forecast['ds'].dt.weekday >= 5))]

    if not prediction1.empty:
        yhat_period = prediction1['yhat'].values[0]
        return yhat_period
    else:
        return None

    # Converter a coluna 'ds' para formato de data, se ainda não estiver em formato de data
    # df1['ds'] = pd.to_datetime(df1['ds'])

    # fig, ax = plt.subplots(figsize=(16, 8))

    # fcst_t = np.array(fcst['ds'].dt.to_pydatetime())
    # ax.plot(np.array(m.history['ds'].dt.to_pydatetime()), m.history['y'], 'k.')

    # ax.plot(df1['ds'], df1['y'])

    # ax.set_title('Preço de Fechamento ao Longo do Tempo')
    # ax.set_xlabel('Data')
    # ax.set_ylabel('Preço de Fechamento')

    # Configurar o formato do eixo x para exibir apenas o ano

    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # ax.xaxis.set_major_locator(mdates.YearLocator())
    # fig.autofmt_xdate()  # Rotacionar os anos para melhor visualização

    # Mostrar o gráfico no Streamlit
    # st.pyplot(fig)


st.title('📈 Projeção do Índice Bovespa')

"""
Este data app usa a Biblioteca open-source Prophet para automaticamente gerar valores futuros de previsão de um dataset importado
Você poderá visualizar as projeções do índice Bovespa para o período de 01/01/2024 a 31/01/2024 😵

Created by Henrique José Itzcovici

Code available here: https://github.com/henitz/streamlit_bovespa
"""

"""
### Passo 1: Importar dados
"""
df = pd.DataFrame(columns=['Data'])  # Inicializa um DataFrame vazio

uploaded_file = st.file_uploader(
    'Importar o dados da série csv aqui, posteriormente as colunas serão nomeadas ds e y. A entrada de dados para o Prophet sempre deve ser com as colunas: ds e y. O ds(datestamp) coluna deve ter o formato esperado pelo Pandas, idealmente YYYY-MM-DD para data ou YYYY-MM-DD HH:MM:SS para timestamp. A coluna y deve ser numérica e representa a medida que queremos estimar',
    type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if not df.empty:
        # Adiciona uma nova coluna 'ID' com números sequenciais
        df.reset_index(drop=False, inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)
        df = df.rename(columns={'Data': 'ds', 'Último': 'y'})
        df.set_index('id', inplace=True)

        df['ds'] = pd.to_datetime(df['ds'], format='%d.%m.%Y')
        # Lista das colunas a serem removidas
        colunas_para_remover = ['Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%']

        # Remover as colunas especificadas do DataFrame
        df = df.drop(columns=colunas_para_remover)
        st.text(df)
        modelo(df)
        st.text("Data")
        flag = False
        data = st.slider('Data', 1, 31, 1)
        if data <= 9:
            data2 = '2024-01-0' + str(data)
        else:
            data2 = '2024-01-' + str(data)

        btn = st.button("predict")

        if btn:
            x = prevendo(df, data2)
            if x is None:
                st.write(f"A data {data2} não está disponível nas previsões ou é feriado/final de semana.")
            else:
                rounded_x = round(x, 2)
                st.write(f"Valor previsto para {data2}: {rounded_x}")
        flag = True
        prevendo(df, data)

