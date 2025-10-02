
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import traceback
import os
import sys

# Imports LangChain/LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Imports para ferramentas personalizadas
sys.path.append('..')
from tools.data_analysis_tools import DataAnalysisTools
from tools.visualization_tools import VisualizationTools
from memory.conversation_memory import ConversationMemory
from utils.data_processor import DataProcessor
from config.settings import GEMINI_CONFIG, AGENT_CONFIG, SYSTEM_PROMPTS

class EDAAgent:
    """
    Agente Autônomo de Análise Exploratória de Dados (EDA) 
    Integrado com Gemini 2.5 Flash e LangChain
    """

    def __init__(self, api_key: str, temperature: float = 0.1, max_iterations: int = 5):
        """
        Inicializa o agente EDA

        Args:
            api_key: Chave da API do Google
            temperature: Temperatura para geração de respostas
            max_iterations: Máximo de iterações por análise
        """
        self.api_key = api_key
        self.temperature = temperature
        self.max_iterations = max_iterations

        # Inicializar LLM
        self._setup_llm()

        # Ferramentas e componentes
        self.data_tools = DataAnalysisTools()
        self.viz_tools = VisualizationTools()
        self.memory = ConversationMemory()
        self.data_processor = DataProcessor()

        # Estado do agente
        self.dataframe = None
        self.dataframe_info = {}
        self.current_session_id = None

    def _setup_llm(self):
        """Configura o modelo LLM Gemini"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=self.temperature,
                google_api_key=self.api_key,
                convert_system_message_to_human=True
            )
            print("✅ LLM Gemini 2.5 Flash configurado com sucesso")
        except Exception as e:
            print(f"❌ Erro ao configurar LLM: {e}")
            raise

    def load_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Carrega o dataframe e extrai informações básicas

        Args:
            df: DataFrame pandas para análise

        Returns:
            Informações básicas do dataset
        """
        try:
            self.dataframe = df.copy()

            # Gerar informações básicas do dataset
            self.dataframe_info = self.data_processor.extract_dataframe_info(df)

            # Iniciar nova sessão
            self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            print(f"✅ Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
            return self.dataframe_info

        except Exception as e:
            print(f"❌ Erro ao carregar dataset: {e}")
            raise

    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Realiza análise baseada na pergunta do usuário

        Args:
            question: Pergunta sobre os dados

        Returns:
            Resultado da análise com texto, gráficos e cálculos
        """
        if self.dataframe is None:
            raise ValueError("Dataset não carregado. Use load_dataframe() primeiro.")

        print(f"🔍 Iniciando análise: {question}")

        try:
            # Criar agente pandas para análise
            pandas_agent = create_pandas_dataframe_agent(
                self.llm,
                self.dataframe,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True,
                max_iterations=self.max_iterations
            )

            # Contexto adicional baseado no histórico
            context = self._build_context(question)

            # Construir prompt melhorado
            enhanced_prompt = self._build_enhanced_prompt(question, context)

            # Executar análise
            analysis_result = pandas_agent.invoke(enhanced_prompt)

            # Processar resultado
            result = self._process_analysis_result(question, analysis_result)

            # Gerar visualizações se necessário
            charts = self._generate_visualizations(question, result)

            # Consolidar resultado final
            final_result = {
                "question": question,
                "analysis": result.get("output", ""),
                "calculations": self._extract_calculations(result),
                "charts": charts,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.current_session_id
            }

            print("✅ Análise concluída com sucesso")
            return final_result

        except Exception as e:
            error_msg = f"Erro na análise: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()

            return {
                "question": question,
                "analysis": f"Não foi possível completar a análise. {error_msg}",
                "calculations": {},
                "charts": [],
                "timestamp": datetime.now().isoformat(),
                "session_id": self.current_session_id,
                "error": True
            }

    def _build_context(self, question: str) -> str:
        """Constrói contexto baseado no histórico de conversas"""
        context_parts = [
            f"INFORMAÇÕES DO DATASET:",
            f"- Linhas: {self.dataframe_info.get('shape', {}).get('rows', 0)}",
            f"- Colunas: {self.dataframe_info.get('shape', {}).get('columns', 0)}",
            f"- Colunas numéricas: {', '.join(self.dataframe_info.get('numeric_columns', []))}",
            f"- Colunas categóricas: {', '.join(self.dataframe_info.get('categorical_columns', []))}",
            f"- Valores ausentes: {self.dataframe_info.get('missing_values_summary', {})}",
        ]

        # Adicionar histórico de perguntas anteriores
        recent_interactions = self.memory.get_recent_interactions(5)
        if recent_interactions:
            context_parts.append("\nINTERAÇÕES ANTERIORES:")
            for interaction in recent_interactions:
                context_parts.append(f"- P: {interaction['question']}")

        return "\n".join(context_parts)

    def _build_enhanced_prompt(self, question: str, context: str) -> str:
        """Constrói prompt aprimorado para o agente"""
        prompt = f"""
{SYSTEM_PROMPTS['data_analyst']}

CONTEXTO DO DATASET:
{context}

PERGUNTA DO USUÁRIO: {question}

INSTRUÇÕES ESPECÍFICAS:
1. Analise os dados de forma sistemática e detalhada
2. Use Python/pandas para cálculos precisos
3. Identifique padrões, tendências e insights relevantes
4. Forneça números específicos e evidências
5. Se necessário, sugira visualizações complementares
6. Seja claro e objetivo nas conclusões

Por favor, forneça uma análise completa e fundamentada.
"""
        return prompt

    def _process_analysis_result(self, question: str, raw_result: Any) -> Dict[str, Any]:
        """Processa o resultado bruto da análise"""
        try:
            if isinstance(raw_result, dict):
                return raw_result
            elif hasattr(raw_result, 'content'):
                return {"output": raw_result.content}
            else:
                return {"output": str(raw_result)}
        except Exception as e:
            return {"output": f"Erro ao processar resultado: {e}"}

    def _extract_calculations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai cálculos e métricas do resultado"""
        calculations = {}
        output = result.get("output", "")

        # Aqui você pode implementar lógica para extrair métricas específicas
        # Por exemplo, procurar por números, percentuais, etc.

        return calculations

    def _generate_visualizations(self, question: str, result: Dict[str, Any]) -> List[str]:
        """Gera visualizações quando apropriado"""
        charts = []

        try:
            # Determinar se a pergunta requer visualização
            viz_keywords = [
                'gráfico', 'chart', 'plot', 'visualiz', 'histograma', 
                'scatter', 'correlação', 'distribuição', 'boxplot'
            ]

            needs_visualization = any(keyword in question.lower() for keyword in viz_keywords)

            if needs_visualization:
                # Determinar tipo de visualização
                chart_type = self._determine_chart_type(question)

                if chart_type:
                    chart_path = self.viz_tools.create_chart(
                        self.dataframe, 
                        chart_type, 
                        question,
                        self.current_session_id
                    )
                    if chart_path:
                        charts.append(chart_path)

        except Exception as e:
            print(f"⚠️ Erro ao gerar visualização: {e}")

        return charts

    def _determine_chart_type(self, question: str) -> Optional[str]:
        """Determina o tipo de gráfico mais apropriado"""
        question_lower = question.lower()

        chart_mappings = {
            'correlação': 'correlation_matrix',
            'distribuição': 'histogram',
            'histograma': 'histogram',
            'scatter': 'scatter',
            'boxplot': 'boxplot',
            'box plot': 'boxplot',
            'heatmap': 'heatmap',
            'barras': 'bar',
            'bar': 'bar'
        }

        for keyword, chart_type in chart_mappings.items():
            if keyword in question_lower:
                return chart_type

        return 'histogram'  # Default

    def save_interaction(self, question: str, result: Dict[str, Any]):
        """Salva interação na memória do agente"""
        self.memory.save_interaction(
            session_id=self.current_session_id,
            question=question,
            result=result
        )

    def get_final_summary(self) -> str:
        """Gera resumo final de todas as análises da sessão"""
        if not self.memory.memory:
            return "Nenhuma análise foi realizada ainda."

        try:
            # Consolidar todas as interações da sessão atual
            session_interactions = [
                interaction for interaction in self.memory.memory 
                if interaction.get('session_id') == self.current_session_id
            ]

            if not session_interactions:
                return "Nenhuma análise encontrada para a sessão atual."

            # Criar prompt para resumo
            summary_prompt = f"""
{SYSTEM_PROMPTS['summary_generator']}

DADOS DO DATASET:
{json.dumps(self.dataframe_info, indent=2)}

ANÁLISES REALIZADAS:
"""

            for i, interaction in enumerate(session_interactions, 1):
                summary_prompt += f"\n{i}. PERGUNTA: {interaction['question']}"
                summary_prompt += f"\n   RESPOSTA: {interaction['result'].get('analysis', '')[:500]}..."

            summary_prompt += "\n\nGere um resumo executivo consolidado com os principais insights descobertos."

            # Gerar resumo usando LLM
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])

            return summary_response.content

        except Exception as e:
            return f"Erro ao gerar resumo final: {e}"

    def clear_session(self):
        """Limpa a sessão atual"""
        self.dataframe = None
        self.dataframe_info = {}
        self.current_session_id = None
        self.memory.clear_session()
        print("🔄 Sessão limpa")
