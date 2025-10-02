
import os
from dotenv import load_dotenv
from typing import Dict, Any

# Carregar variáveis de ambiente
load_dotenv()

# Configurações da API
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Configurações do Gemini 2.5 Flash
GEMINI_CONFIG = {
    "model_name": "gemini-2.5-flash",
    "temperature": 0.1,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40
}

# Limites da API gratuita do Gemini 2.5 Flash
API_LIMITS = {
    "requests_per_minute": 10,
    "tokens_per_minute": 250000,
    "requests_per_day": 250
}

# Configurações do agente EDA
AGENT_CONFIG = {
    "max_iterations": 5,
    "max_memory_interactions": 50,
    "chart_output_dir": "outputs/charts",
    "report_output_dir": "outputs/reports",
    "supported_chart_types": [
        "histogram", "scatter", "boxplot", "heatmap", 
        "bar", "line", "pie", "violin", "correlation_matrix"
    ]
}

# Configurações de visualização
CHART_CONFIG = {
    "figure_size": (10, 6),
    "dpi": 300,
    "style": "whitegrid",
    "color_palette": "viridis",
    "font_size": 12
}

# Prompts do sistema para diferentes tipos de análise
SYSTEM_PROMPTS = {
    "data_analyst": """
    Você é um especialista em Análise Exploratória de Dados (EDA) com vasta experiência em Python, pandas, matplotlib e seaborn.

    Suas responsabilidades:
    1. Analisar datasets CSV de forma detalhada e sistemática
    2. Identificar padrões, tendências e insights relevantes
    3. Detectar valores ausentes, outliers e anomalias
    4. Criar visualizações informativas quando necessário
    5. Fornecer conclusões claras e acionáveis

    Sempre:
    - Seja preciso e objetivo em suas análises
    - Use linguagem técnica adequada mas compreensível
    - Sugira próximos passos quando relevante
    - Cite métricas específicas e evidências
    """,

    "chart_generator": """
    Você é um especialista em visualização de dados com Python.

    Sua função é gerar código Python para criar gráficos usando matplotlib, seaborn ou plotly baseado na análise solicitada.

    Diretrizes:
    1. Retorne APENAS código Python executável
    2. Use bibliotecas: matplotlib.pyplot, seaborn, pandas
    3. Configure estilo e cores adequadas
    4. Adicione títulos, labels e legendas
    5. Salve o gráfico em arquivo PNG de alta qualidade

    Formato de saída:
    ```python
    # Código Python aqui
    ```
    """,

    "summary_generator": """
    Você é um especialista em síntese de informações e geração de relatórios executivos.

    Sua função é consolidar todas as análises e insights obtidos em um resumo executivo claro e estruturado.

    Inclua:
    1. Resumo executivo dos principais insights
    2. Características gerais do dataset
    3. Descobertas mais importantes
    4. Recomendações e próximos passos
    5. Limitações e considerações técnicas
    """
}

# Configurações de cache e otimização
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_ttl_seconds": 3600,
    "max_dataframe_size_mb": 100,
    "sample_size_for_large_datasets": 10000
}

# Configurações de segurança
SECURITY_CONFIG = {
    "max_file_size_mb": 50,
    "allowed_file_extensions": [".csv"],
    "sanitize_column_names": True,
    "validate_data_types": True
}

def get_config() -> Dict[str, Any]:
    """Retorna todas as configurações consolidadas"""
    return {
        "gemini": GEMINI_CONFIG,
        "agent": AGENT_CONFIG,
        "charts": CHART_CONFIG,
        "prompts": SYSTEM_PROMPTS,
        "performance": PERFORMANCE_CONFIG,
        "security": SECURITY_CONFIG,
        "api_limits": API_LIMITS
    }

def validate_api_key() -> bool:
    """Valida se a chave da API está configurada"""
    return bool(GEMINI_API_KEY and len(GEMINI_API_KEY) > 10)

# Criar diretórios necessários
def setup_directories():
    """Cria diretórios necessários para o projeto"""
    import os
    os.makedirs(AGENT_CONFIG["chart_output_dir"], exist_ok=True)
    os.makedirs(AGENT_CONFIG["report_output_dir"], exist_ok=True)
    os.makedirs("data", exist_ok=True)
