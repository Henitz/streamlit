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

import Model
import prevel_model

# Ignorar os FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# selected_date = '2024-01-25'  # Replace this with the selected date
# selected_time = pd.Timestamp('00:00:00').time()

from Model import modelo
from prevel_model import prevendo

import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st

st.title('📈 Projeção do Índice Bovespa')

"""
Este data app usa a Biblioteca open-source Prophet para automaticamente gerar valores futuros de previsão de um dataset importado
Você poderá visualizar as projeções do índice Bovespa para o período de 01/01/2024 a 31/01/2024 😵

criado por Henrique José Itzcovici

Código avaliável em: https://github.com/Henitz/streamlit
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

    if not df.empty and 'Data' in df.columns and 'Último' in df.columns:
        # Renomear colunas para 'ds' e 'y'
        df = df.rename(columns={'Data': 'ds', 'Último': 'y'})
        df['ds'] = pd.to_datetime(df['ds'], format='%d.%m.%Y')

        # Outras operações no DataFrame, como remover colunas indesejadas
        colunas_para_remover = ['Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%']
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

    else:
        st.warning("O arquivo não foi carregado corretamente ou não possui as colunas esperadas.")

if uploaded_file is not None and not df.empty and 'ds' in df.columns:
    """
    ### Passo 2: Modelo
    """
    Model.modelo(df, data_selecionada, hora_selecionada)
    st.markdown(
        """
        <div style="border: 2px solid black; border-radius: 5px; padding: 20px; width: 80%;">
    <p style="text-align: justify;"><strong>MAE (Mean Absolute Error)</strong>: Representa a média das diferenças absolutas entre as previsões e os valores reais. Indica o quão perto as previsões estão dos valores reais, sem considerar a direção do erro.</p>
    <p style="text-align: justify;"><strong>MSE (Mean Squared Error)</strong>: É a média das diferenças quadradas entre as previsões e os valores reais. Penaliza erros maiores mais significativamente que o MAE, devido ao termo quadrático, o que torna o MSE mais sensível a outliers.</p>
    <p style="text-align: justify;"><strong>RMSE (Root Mean Squared Error)</strong>: É a raiz quadrada do MSE. Apresenta o mesmo tipo de informação que o MSE, mas na mesma unidade que os dados originais, o que facilita a interpretação.</p>
    <p style="text-align: justify;"><strong>MAPE (Mean Absolute Percentage Error)</strong>: É uma métrica usada para avaliar a precisão de um modelo de previsão em relação ao tamanho dos erros em termos percentuais. Essa métrica calcula a média dos valores absolutos dos erros percentuais entre os valores reais e os valores previstos.</p>
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="border: 2px solid black; border-radius: 5px; padding: 10px;"> 
            <h6>MAE</h6>
            <p style="text-align: justify;">O valor do MAE, como qualquer métrica de erro, depende muito do contexto e da escala dos dados que você está considerando. No contexto do mercado de ações, um MAE de 12,06 pontos pode ser considerado alto ou baixo dependendo do valor médio dos índices ou ativos que você está analisando.</p>
            <p style="text-align: justify;">
            Se o índice ou ativo em questão tem uma faixa de valores geralmente baixa (por exemplo, entre 100 e 200 pontos), um MAE de 12,06 pode ser considerado significativo, representando uma porcentagem considerável dessa faixa.
            </p>
            <p style="text-align: justify;">
            Por outro lado, se o índice ou ativo tem valores muito mais altos (por exemplo, entre 1000 e 2000 pontos), um MAE de 12,06 pode ser relativamente pequeno.
            </p>
            <p style="text-align: justify;">
            O importante é contextualizar esse valor em relação à escala dos dados que está sendo analisada e considerar como ele se compara a outros modelos ou análises similares. Em geral, um MAE mais baixo indica um melhor desempenho do modelo em prever os valores reais.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="border: 2px solid black; border-radius: 5px; padding: 10px;">
            <h6><strong>MAPE</strong></h6> 
            <p style="text-align: justify;">Para a bolsa de valores, 11,65% é um valor razoável?</p>
            <p style="text-align: justify;">
            Para a bolsa de valores, um MAPE de 11,65% pode ser considerado relativamente alto em muitos contextos devido à sensibilidade e à volatilidade desse ambiente. No entanto, no mundo da previsão financeira e de ações, avaliar se um MAPE de 11,65% é considerado aceitável ou não depende de diversos fatores:
            </p>
            <ul style="text-align: justify;">
                <li><strong>Horizonte de Tempo:</strong> O MAPE pode variar dependendo do horizonte de tempo das previsões. Em curtos períodos de tempo, como previsões intra-diárias, um MAPE de 11,65% seria geralmente considerado alto. Já em previsões de longo prazo, talvez seja mais aceitável.</li>
                <li><strong>Instrumento Financeiro:</strong> Diferentes tipos de ativos (ações, commodities, moedas) podem ter comportamentos diferentes. Algumas ações podem ser mais voláteis e imprevisíveis do que outras.</li>
                <li><strong>Estratégia de Negociação:</strong> O MAPE aceitável pode variar de acordo com a estratégia de negociação. Para um investidor de longo prazo, um MAPE mais alto pode ser tolerável, enquanto para traders de curto prazo, pode ser considerado menos aceitável.</li>
                <li><strong>Comparação com Referências:</strong> É útil comparar o MAPE obtido com o desempenho de outros modelos de previsão ou com benchmarks do mercado financeiro para avaliar sua eficácia relativa.</li>
                <li><strong>Consequências Financeiras:</strong> Avalie as consequências financeiras do MAPE. Mesmo que 11,65% pareça alto, se as previsões permitirem tomar decisões lucrativas ou reduzir perdas, pode ser aceitável.</li>
            </ul>
            <p style="text-align: justify;">
            Em geral, para muitos investidores e analistas da bolsa de valores, um MAPE de 11,65% poderia ser considerado relativamente alto, especialmente se a precisão das previsões for crucial para estratégias de negociação específicas. Contudo, é crucial contextualizar o MAPE dentro das especificidades do mercado financeiro e considerar outros indicadores e métricas ao avaliar a eficácia das previsões.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
    <div style="border: 2px solid black; border-radius: 5px; padding: 10px; text-align: justify;">
    <h6><strong>RMSE</strong></h6>
    <p>Em muitos casos envolvendo previsão na bolsa de valores, um RMSE de 11,65 pode ser considerado alto, especialmente se estiver lidando com a previsão de preços de ações individuais ou ativos específicos. No contexto financeiro, pequenas diferenças nas previsões podem ter um impacto significativo nos resultados e nas decisões de investimento.</p>
    <p>Um RMSE de 11,65 indicaria que, em média, as previsões estão a cerca de 11,65 unidades de distância dos valores reais. Para muitos investidores e analistas financeiros, essa margem de erro pode ser considerada grande, especialmente ao lidar com investimentos de curto prazo ou estratégias de trading onde a precisão é crucial.</p>
    <p>Portanto, para previsões na bolsa de valores, é comum buscar valores de erro menores, indicando uma maior precisão nas previsões. Um RMSE de 11,65 pode ser visto como relativamente alto, sugerindo a necessidade de melhorias no modelo para tornar as previsões mais precisas e confiáveis.</p>
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
        x = prevendo(df, data2, flag)
        if x is None:
            st.write(f"A data {data2} não está disponível nas previsões ou é feriado/final de semana.")
        else:
            rounded_x = round(x, 3)
            st.write(f"Valor previsto para {data2}: {rounded_x}")
    flag = True
    prevel_model.prevendo(df, data)
