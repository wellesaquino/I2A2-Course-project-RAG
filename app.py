
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import sys
sys.path.append('src')

from agents.eda_agent import EDAAgent
from config.settings import GEMINI_API_KEY, AGENT_CONFIG

st.set_page_config(
    page_title="🤖 Agente EDA Autônomo", 
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("🤖 Agente EDA Autônomo")
    st.markdown("**Análise Exploratória de Dados Inteligente com Gemini 2.5 Flash**")

    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")

        # Upload do arquivo CSV
        uploaded_file = st.file_uploader(
            "📁 Carregar arquivo CSV", 
            type=['csv'],
            help="Faça upload do seu dataset em formato CSV"
        )

        # Configurações do agente
        st.subheader("🎛️ Parâmetros do Agente")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_iterations = st.slider("Max Iterações", 1, 10, 5)

        # Perguntas pré-definidas
        st.subheader("❓ Perguntas Sugeridas")
        predefined_questions = [
            "Análise geral do dataset",
            "Detectar valores ausentes e outliers", 
            "Análise de correlações",
            "Distribuição das variáveis numéricas",
            "Análise de variáveis categóricas",
            "Principais insights e conclusões"
        ]

        selected_question = st.selectbox(
            "Escolha uma pergunta:",
            ["Pergunta personalizada"] + predefined_questions
        )

    # Área principal
    if uploaded_file is not None:
        # Carregar e mostrar preview dos dados
        try:
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

            with st.expander("👀 Preview dos Dados"):
                st.dataframe(df.head(10))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Linhas", df.shape[0])
                with col2:
                    st.metric("Colunas", df.shape[1]) 
                with col3:
                    st.metric("Memoria (MB)", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}")

            # Inicializar agente EDA
            if 'eda_agent' not in st.session_state:
                st.session_state.eda_agent = EDAAgent(
                    api_key=GEMINI_API_KEY,
                    temperature=temperature,
                    max_iterations=max_iterations
                )
                st.session_state.eda_agent.load_dataframe(df)

            # Área de perguntas
            st.subheader("💬 Faça sua pergunta sobre os dados")

            if selected_question != "Pergunta personalizada":
                question = selected_question
                st.info(f"Pergunta selecionada: {question}")
            else:
                question = st.text_input(
                    "Digite sua pergunta:",
                    placeholder="Ex: Quais são as principais correlações no dataset?"
                )

            if st.button("🚀 Analisar", type="primary") and question:
                with st.spinner("🔄 Analisando dados... Isso pode levar alguns minutos."):
                    try:
                        # Executar análise
                        result = st.session_state.eda_agent.analyze(question)

                        # Exibir resultados
                        st.subheader("📊 Resultados da Análise")

                        # Resposta textual
                        if result.get('analysis'):
                            st.markdown("### 📝 Análise")
                            st.write(result['analysis'])

                        # Gráficos gerados
                        if result.get('charts'):
                            st.markdown("### 📈 Visualizações")
                            for i, chart_path in enumerate(result['charts']):
                                if os.path.exists(chart_path):
                                    st.image(chart_path, caption=f"Gráfico {i+1}")

                        # Cálculos e métricas
                        if result.get('calculations'):
                            st.markdown("### 🔢 Cálculos")
                            st.json(result['calculations'])

                        # Salvar na memória do agente
                        st.session_state.eda_agent.save_interaction(question, result)

                    except Exception as e:
                        st.error(f"❌ Erro na análise: {str(e)}")

        except Exception as e:
            st.error(f"❌ Erro ao carregar arquivo: {str(e)}")

    else:
        st.info("👆 Faça upload de um arquivo CSV para começar a análise.")

        # Exemplo de dataset
        st.markdown("### 🎯 Como usar:")
        st.markdown("""
        1. **Upload**: Carregue seu arquivo CSV
        2. **Perguntas**: Escolha uma pergunta pré-definida ou faça uma personalizada
        3. **Análise**: O agente irá analisar automaticamente seus dados
        4. **Visualizações**: Gráficos serão gerados quando relevante
        """)

if __name__ == "__main__":
    main()
