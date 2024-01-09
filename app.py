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
from prophet.diagnostics import performance_metrics

# Ignorar os FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

selected_date = '2024-01-25'  # Replace this with the selected date
selected_time = pd.Timestamp('00:00:00').time()


def modelo(df1):
    import pandas as pd
    import numpy as np
    from prophet import Prophet
    import plotly.graph_objs as go
    import plotly.express as px
    import streamlit as st

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
    # Criar campo 'is_weekend' com 0 e 1 - significando fim de semana
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

    # Criando o gráfico com Plotly
    fig = go.Figure()

    # Adicionando os dados do histórico
    fig.add_trace(go.Scatter(x=history_ds, y=m.history['y'], mode='markers', name='Histórico'))

    # Adicionando os dados da previsão
    fig.add_trace(go.Scatter(x=fcst_t, y=forecast['yhat'], mode='lines', name='Previsão'))

    # Adicionando a faixa de intervalo
    fig.add_trace(go.Scatter(
        x=np.concatenate([fcst_t, fcst_t[::-1]]),
        y=np.concatenate([forecast['yhat_lower'], forecast['yhat_upper'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Intervalo de Confiança'
    ))

    # Personalizando o layout do gráfico
    fig.update_layout(
        xaxis_title='Data',
        yaxis_title='Valores',
        title='Previsão com Prophet e Intervalo de Confiança'
    )

    # Exibindo o gráfico no Streamlit
    st.plotly_chart(fig)

    # Mostrando

    # Calculando métricas de desempenho
    # df_cv = performance_metrics(df)
    # st.write(df_cv)
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

    if data_selecionada is not None and hora_selecionada is not None:
        # Convert the selected date to the DataFrame's format
        data_formatada_interna = pd.to_datetime(data_selecionada).strftime('%Y-%m-%d')

        # Create a datetime combining the selected date with the default time
        data_hora_interna = pd.to_datetime(data_formatada_interna + ' ' + str(hora_selecionada))

        # Filter the DataFrame based on the selected date and time
        df_filtrado_interno = forecast[forecast['ds'] == data_hora_interna]
        st.dataframe(df_filtrado_interno)
    else:
        st.warning("Não há dados para a data selecionada.")


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


st.title('📈 Projeção do Índice Bovespa')

"""
Este data app usa a Biblioteca open-source Prophet para automaticamente gerar valores futuros de previsão de um dataset importado
Você poderá visualizar as projeções do índice Bovespa para o período de 01/01/2024 a 31/01/2024 😵

criado por Henrique José Itzcovici

Código avaliável em: https://github.com/henitz/streamlit
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

        st.dataframe(df.style.set_table_attributes('style="height: 50px; overflow: auto;"'))

        if 'ds' in df.columns:
            data_padrao = df['ds'].min()
            hora_padrao = pd.Timestamp('00:00:00').time()  # Hora padrão como 00:00:00
            data_minima = df['ds'].min()  # Data mínima do DataFrame
            data_maxima = df['ds'].max()  # Data máxima do DataFrame

            data_selecionada = st.sidebar.date_input("Selecione uma data", value=data_padrao, min_value=data_minima,
                                                     max_value=data_maxima)
            hora_selecionada = st.sidebar.time_input("Selecione um horário", value=hora_padrao)

            if data_selecionada:
                # Convertendo a data selecionada para o formato do DataFrame
                data_selecionada_formatada = pd.to_datetime(data_selecionada).strftime('%Y-%m-%d')

                # Criar um datetime combinando a data selecionada com a hora padrão
                data_hora_selecionada = pd.to_datetime(data_selecionada_formatada + ' ' + str(hora_selecionada))

                # Filtrar o DataFrame com base na data e hora selecionadas
                df_filtrado = df[df['ds'] == data_hora_selecionada]
                st.dataframe(df_filtrado)
            else:
                st.warning("Não há dados para a data selecionada.")
        else:
            st.warning("A coluna 'ds' não está presente no DataFrame.")

        """
        ### Passo 2: Modelo
        """

        modelo(df)

        st.markdown(
            """
            <div style="border: 2px solid black; border-radius: 5px; padding: 10px;">
                <p><strong>MAE (Mean Absolute Error)</strong>: Representa a média das diferenças absolutas entre as previsões e os valores reais. Indica o quão perto as previsões estão dos valores reais, sem considerar a direção do erro.</p>
                <p><strong>MSE (Mean Squared Error)</strong>: É a média das diferenças quadradas entre as previsões e os valores reais. Penaliza erros maiores mais significativamente que o MAE, devido ao termo quadrático, o que torna o MSE mais sensível a outliers.</p>
                <p><strong>RMSE (Root Mean Squared Error)</strong>: É a raiz quadrada do MSE. Apresenta o mesmo tipo de informação que o MSE, mas na mesma unidade que os dados originais, o que facilita a interpretação.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        """
        ### Passo 3: Previsão no Intervalo 01/01/2024 a 31/01/2024
        """

        flag = False
        data = st.slider('Data', 1, 31, 1)
        if data <= 9:
            data2 = '2024-01-0' + str(data)
        else:
            data2 = '2024-01-' + str(data)

        btn = st.button("Previsão")

        if btn:
            x = prevendo(df, data2)
            if x is None:
                st.write(f"A data {data2} não está disponível nas previsões ou é feriado/final de semana.")
            else:
                rounded_x = round(x, 3)
                st.write(f"Valor previsto para {data2}: {rounded_x}")
        flag = True
        prevendo(df, data)
