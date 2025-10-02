# 🤖 Agente EDA Autônomo

**Análise Exploratória de Dados Inteligente com Gemini 2.5 Flash**

Este projeto implementa um agente autônomo de Análise Exploratória de Dados (EDA) que utiliza LangChain, LangGraph e o modelo Gemini 2.5 Flash para analisar datasets CSV de forma completamente automática.

## 📋 Características

### 🎯 Funcionalidades Principais
- **Upload CSV Interativo**: Interface Streamlit para carregar datasets
- **Análise Automática**: O agente compreende perguntas em linguagem natural
- **Visualizações Inteligentes**: Gera gráficos automaticamente quando relevante
- **Memória Contextual**: Lembra de análises anteriores na mesma sessão
- **Relatórios PDF**: Gera relatórios profissionais com todas as descobertas

### 🧠 Capacidades de Análise
- Estatísticas descritivas completas
- Detecção de valores ausentes e outliers
- Análise de correlações entre variáveis
- Distribuições e padrões nos dados
- Análise de variáveis categóricas
- Identificação de problemas de qualidade

### 📊 Tipos de Visualização
- Histogramas e distribuições
- Scatter plots com linha de tendência
- Heatmaps de correlação
- Boxplots para detecção de outliers
- Gráficos de barras para categóricas
- Dashboards de visão geral

## 🚀 Instalação e Configuração

### 1. Pré-requisitos
- Python 3.8+
- Chave da API do Google (Gemini)

### 2. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/agente-eda-autonomo.git
cd agente-eda-autonomo
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente
```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar com sua chave da API
nano .env
```

No arquivo `.env`, configure:
```env
GOOGLE_API_KEY=sua_chave_gemini_aqui
```

### 5. Obter Chave da API Gemini

1. Acesse [Google AI Studio](https://ai.google.dev/)
2. Faça login com sua conta Google
3. Crie uma nova chave de API
4. Copie a chave para o arquivo `.env`

**Limites da API Gratuita (Gemini 2.5 Flash):**
- 10 requisições por minuto
- 250.000 tokens por minuto  
- 250 requisições por dia

## 🎮 Como Usar

### 1. Executar a Aplicação
```bash
streamlit run app.py
```

### 2. Carregar Dataset
- Acesse a interface no navegador (geralmente `http://localhost:8501`)
- Use o sidebar para fazer upload do arquivo CSV
- Visualize o preview dos dados

### 3. Fazer Perguntas
Escolha uma pergunta pré-definida ou escreva a sua própria:

**Exemplos de perguntas:**
- "Análise geral do dataset"
- "Quais são as principais correlações?"
- "Existem outliers nos dados?"
- "Como estão distribuídas as variáveis numéricas?"
- "Há valores ausentes significativos?"

### 4. Interpretar Resultados
- **Análise Textual**: Interpretação detalhada pelo agente
- **Visualizações**: Gráficos gerados automaticamente
- **Cálculos**: Métricas e estatísticas relevantes

### 5. Gerar Relatório Final
- Clique em "Gerar Relatório PDF Completo"
- Download do relatório com todas as análises

## 🏗️ Arquitetura do Sistema

### Estrutura de Diretórios
```
agente_eda_autonomo/
├── app.py                    # Aplicação Streamlit principal
├── src/
│   ├── agents/
│   │   └── eda_agent.py     # Agente principal EDA
│   ├── tools/
│   │   ├── data_analysis_tools.py    # Ferramentas de análise
│   │   └── visualization_tools.py    # Ferramentas de visualização
│   ├── memory/
│   │   └── conversation_memory.py    # Sistema de memória
│   ├── utils/
│   │   ├── data_processor.py         # Processamento de dados
│   │   └── pdf_generator.py          # Geração de relatórios
│   └── config/
│       └── settings.py      # Configurações do sistema
├── outputs/
│   ├── charts/              # Gráficos gerados
│   └── reports/             # Relatórios PDF
├── requirements.txt         # Dependências
└── .env.example            # Exemplo de configuração
```

### Componentes Principais

#### 🤖 EDAAgent (`src/agents/eda_agent.py`)
- Orquestra todo o processo de análise
- Integra LangChain com Gemini 2.5 Flash
- Gerencia contexto e memória
- Coordena ferramentas especializadas

#### 🔧 Ferramentas Especializadas
- **DataAnalysisTools**: Análises estatísticas e detecção de padrões
- **VisualizationTools**: Geração automática de gráficos
- **ConversationMemory**: Sistema de memória contextual
- **DataProcessor**: Limpeza e preparação de dados

#### 📄 Geração de Relatórios
- **PDFGenerator**: Criação de relatórios profissionais
- Layout estruturado com sumário executivo
- Integração de gráficos e análises textuais

## ⚙️ Configuração Avançada

### Parâmetros do Agente
```python
# Configurações em src/config/settings.py
AGENT_CONFIG = {
    "max_iterations": 5,           # Máximo de iterações por análise
    "max_memory_interactions": 50, # Máximo de interações na memória
    "temperature": 0.1,            # Criatividade do modelo (0-1)
}
```

### Tipos de Gráficos Suportados
- `histogram`: Distribuições de variáveis numéricas
- `scatter`: Relações entre duas variáveis
- `boxplot`: Detecção de outliers
- `heatmap`: Correlações e padrões
- `bar`: Distribuições categóricas
- `correlation_matrix`: Matriz de correlação completa

### Personalização de Prompts
Os prompts do sistema podem ser customizados em `src/config/settings.py`:

```python
SYSTEM_PROMPTS = {
    "data_analyst": "Prompt para análise de dados...",
    "chart_generator": "Prompt para geração de gráficos...",
    "summary_generator": "Prompt para resumos..."
}
```

## 🔍 Exemplos de Uso

### Análise de Dataset de Vendas
```
Pergunta: "Analise as correlações entre preço, quantidade e desconto"
Resultado: Matriz de correlação + scatter plots + insights textuais
```

### Detecção de Anomalias
```
Pergunta: "Existem outliers nas variáveis numéricas?"
Resultado: Boxplots + estatísticas IQR + lista de outliers
```

### Qualidade dos Dados
```
Pergunta: "Qual a qualidade geral dos dados?"
Resultado: Relatório de valores ausentes + score de qualidade + recomendações
```

## 🚧 Limitações e Considerações

### Limitações da API Gratuita
- **Rate Limits**: 10 req/min, 250 req/dia
- **Contexto**: Limitado por tokens disponíveis
- **Complexidade**: Análises muito complexas podem falhar

### Recomendações de Uso
- **Tamanho do Dataset**: Idealmente < 100MB para melhor performance
- **Número de Colunas**: < 50 colunas para visualizações otimizadas
- **Sessões**: Reinicie para datasets muito diferentes

### Segurança
- Nunca exponha sua chave da API
- Use `.env` para variáveis sensíveis
- Valide dados antes do upload

## 🛠️ Desenvolvimento e Contribuição

### Executar Testes
```bash
pytest tests/ -v
```

### Adicionar Novas Funcionalidades

1. **Nova Ferramenta de Análise**:
   - Adicione em `src/tools/`
   - Integre no `EDAAgent`
   - Teste com dataset exemplo

2. **Novo Tipo de Visualização**:
   - Implemente em `VisualizationTools`
   - Adicione lógica de detecção no agente
   - Configure prompts apropriados

3. **Melhorias no Relatório**:
   - Modifique `PDFGenerator`
   - Adicione novas seções
   - Teste formatação

### Estrutura de Contribuição
```bash
# Fork do repositório
git checkout -b feature/nova-funcionalidade
git commit -m "Adiciona nova funcionalidade X"
git push origin feature/nova-funcionalidade
# Abrir Pull Request
```

## 📚 Recursos Adicionais

### Documentação
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Gemini API Reference](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Datasets de Exemplo
- [Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Boston Housing](https://www.kaggle.com/c/boston-housing)

## 🐛 Solução de Problemas

### Erro de API Key
```
❌ Erro ao configurar LLM: 401 Unauthorized
```
**Solução**: Verifique se a `GOOGLE_API_KEY` está correta no `.env`

### Rate Limit Excedido
```
❌ Erro na análise: 429 Too Many Requests
```
**Solução**: Aguarde alguns minutos antes de fazer nova requisição

### Arquivo CSV Não Carrega
```
❌ Erro ao carregar arquivo: UnicodeDecodeError
```
**Solução**: Certifique-se de que o arquivo está em UTF-8

### Gráficos Não Aparecem
```
⚠️ Erro ao gerar visualização
```
**Solução**: Verifique se há colunas numéricas suficientes no dataset

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🤝 Contribuidores

- **Desenvolvedor Principal**: Especialista em Python, LangChain e Agentes IA
- **Framework**: LangChain + LangGraph + Gemini 2.5 Flash
- **Inspiração**: Automação de EDA e democratização de análise de dados

## 📞 Suporte

Para dúvidas, sugestões ou problemas:
- Abra uma issue no GitHub
- Consulte a documentação dos frameworks utilizados
- Verifique os exemplos na pasta `examples/`

---

**🚀 Automatize suas análises de dados com inteligência artificial!**

*Desenvolvido com ❤️ usando LangChain, LangGraph e Gemini 2.5 Flash*
