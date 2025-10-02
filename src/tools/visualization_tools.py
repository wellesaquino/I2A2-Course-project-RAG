
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VisualizationTools:
    """
    Ferramentas especializadas para geração automática de visualizações
    """

    def __init__(self):
        # Configurar estilo padrão
        plt.style.use('seaborn-v0_8')
        sns.set_palette("viridis")
        self.figure_size = (10, 6)
        self.dpi = 300

        # Criar diretório de saída se não existir
        self.output_dir = "outputs/charts"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_chart(self, df: pd.DataFrame, chart_type: str, context: str, session_id: str) -> Optional[str]:
        """
        Cria gráfico baseado no tipo e contexto

        Args:
            df: DataFrame para visualização
            chart_type: Tipo de gráfico
            context: Contexto da pergunta
            session_id: ID da sessão

        Returns:
            Caminho do arquivo de imagem gerado
        """
        try:
            if chart_type == "correlation_matrix":
                return self._create_correlation_heatmap(df, session_id)
            elif chart_type == "histogram":
                return self._create_histogram(df, context, session_id)
            elif chart_type == "scatter":
                return self._create_scatter_plot(df, context, session_id)
            elif chart_type == "boxplot":
                return self._create_boxplot(df, context, session_id)
            elif chart_type == "bar":
                return self._create_bar_chart(df, context, session_id)
            elif chart_type == "heatmap":
                return self._create_heatmap(df, session_id)
            else:
                return self._create_overview_charts(df, session_id)

        except Exception as e:
            print(f"❌ Erro ao criar gráfico {chart_type}: {e}")
            return None

    def _create_correlation_heatmap(self, df: pd.DataFrame, session_id: str) -> str:
        """Cria heatmap de correlação"""
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return None

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Calcular correlação
        correlation_matrix = numeric_df.corr()

        # Criar heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .8},
            ax=ax
        )

        ax.set_title('Matriz de Correlação entre Variáveis Numéricas', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Salvar
        filename = f"correlation_heatmap_{session_id}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def _create_histogram(self, df: pd.DataFrame, context: str, session_id: str) -> str:
        """Cria histogramas para variáveis numéricas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return None

        # Determinar layout do subplot
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), dpi=self.dpi)

        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            ax = axes[i]

            # Histogram com KDE
            sns.histplot(
                data=df,
                x=col,
                kde=True,
                ax=ax,
                alpha=0.7
            )

            ax.set_title(f'Distribuição de {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequência')

            # Adicionar estatísticas
            mean_val = df[col].mean()
            median_val = df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Média: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Mediana: {median_val:.2f}')
            ax.legend()

        # Ocultar subplots extras
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Distribuição das Variáveis Numéricas', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Salvar
        filename = f"histogram_{session_id}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def _create_scatter_plot(self, df: pd.DataFrame, context: str, session_id: str) -> str:
        """Cria scatter plot entre duas variáveis numéricas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return None

        # Selecionar as duas primeiras colunas numéricas ou mais correlacionadas
        correlation_matrix = df[numeric_cols].corr()

        # Encontrar o par com maior correlação absoluta
        max_corr = 0
        best_pair = (numeric_cols[0], numeric_cols[1])

        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > max_corr:
                    max_corr = corr_val
                    best_pair = (numeric_cols[i], numeric_cols[j])

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Scatter plot
        sns.scatterplot(
            data=df,
            x=best_pair[0],
            y=best_pair[1],
            alpha=0.6,
            ax=ax
        )

        # Linha de tendência
        sns.regplot(
            data=df,
            x=best_pair[0],
            y=best_pair[1],
            scatter=False,
            color='red',
            ax=ax
        )

        # Correlação
        corr_value = correlation_matrix.loc[best_pair[0], best_pair[1]]

        ax.set_title(f'Scatter Plot: {best_pair[0]} vs {best_pair[1]}\nCorrelação: {corr_value:.3f}', fontweight='bold')
        ax.set_xlabel(best_pair[0])
        ax.set_ylabel(best_pair[1])

        plt.tight_layout()

        # Salvar
        filename = f"scatter_{session_id}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def _create_boxplot(self, df: pd.DataFrame, context: str, session_id: str) -> str:
        """Cria boxplots para detectar outliers"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return None

        # Limitar a 6 colunas para legibilidade
        cols_to_plot = numeric_cols[:6]

        fig, ax = plt.subplots(figsize=(max(8, len(cols_to_plot) * 1.5), 6), dpi=self.dpi)

        # Normalizar dados para comparação
        df_normalized = df[cols_to_plot].copy()
        for col in cols_to_plot:
            df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()

        # Boxplot
        sns.boxplot(data=df_normalized, ax=ax)

        ax.set_title('Boxplot das Variáveis Numéricas (Normalizadas)', fontweight='bold')
        ax.set_xlabel('Variáveis')
        ax.set_ylabel('Valores Normalizados')
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Salvar
        filename = f"boxplot_{session_id}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def _create_bar_chart(self, df: pd.DataFrame, context: str, session_id: str) -> str:
        """Cria gráfico de barras para variáveis categóricas"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) == 0:
            return None

        # Selecionar primeira coluna categórica com cardinalidade razoável
        selected_col = None
        for col in categorical_cols:
            if df[col].nunique() <= 15:  # Máximo 15 categorias
                selected_col = col
                break

        if selected_col is None:
            selected_col = categorical_cols[0]

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Contar valores e pegar top 10
        value_counts = df[selected_col].value_counts().head(10)

        # Gráfico de barras
        sns.barplot(
            x=value_counts.values,
            y=value_counts.index,
            ax=ax,
            palette='viridis'
        )

        ax.set_title(f'Distribuição de {selected_col}', fontweight='bold')
        ax.set_xlabel('Frequência')
        ax.set_ylabel(selected_col)

        # Adicionar valores nas barras
        for i, v in enumerate(value_counts.values):
            ax.text(v + 0.1, i, str(v), va='center')

        plt.tight_layout()

        # Salvar
        filename = f"bar_chart_{session_id}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def _create_heatmap(self, df: pd.DataFrame, session_id: str) -> str:
        """Cria heatmap genérico dos dados"""
        # Usar apenas dados numéricos e amostrar se necessário
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return None

        # Amostrar se muitas linhas
        if len(numeric_df) > 100:
            numeric_df = numeric_df.sample(100)

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Heatmap dos valores
        sns.heatmap(
            numeric_df.T,  # Transpor para ter variáveis nas linhas
            cmap='viridis',
            ax=ax,
            cbar_kws={"shrink": .8}
        )

        ax.set_title('Heatmap dos Dados Numéricos', fontweight='bold')
        ax.set_xlabel('Observações (amostra)')
        ax.set_ylabel('Variáveis')

        plt.tight_layout()

        # Salvar
        filename = f"heatmap_{session_id}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def _create_overview_charts(self, df: pd.DataFrame, session_id: str) -> str:
        """Cria visualização geral do dataset"""
        fig = plt.figure(figsize=(15, 10), dpi=self.dpi)

        # Layout 2x2
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Info básica do dataset
        ax1 = fig.add_subplot(gs[0, 0])
        info_text = f"""
INFORMAÇÕES DO DATASET

Linhas: {df.shape[0]:,}
Colunas: {df.shape[1]}

Tipos de dados:
{df.dtypes.value_counts().to_string()}

Memória: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
"""
        ax1.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Informações Gerais', fontweight='bold')

        # 2. Valores ausentes
        ax2 = fig.add_subplot(gs[0, 1])
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].head(10)

        if not missing_data.empty:
            sns.barplot(x=missing_data.values, y=missing_data.index, ax=ax2)
            ax2.set_title('Valores Ausentes por Coluna', fontweight='bold')
            ax2.set_xlabel('Quantidade')
        else:
            ax2.text(0.5, 0.5, 'Sem valores ausentes', ha='center', va='center')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title('Valores Ausentes', fontweight='bold')

        # 3. Distribuição de tipos
        ax3 = fig.add_subplot(gs[1, 0])
        dtype_counts = df.dtypes.value_counts()

        wedges, texts, autotexts = ax3.pie(
            dtype_counts.values,
            labels=dtype_counts.index,
            autopct='%1.1f%%',
            startangle=90
        )
        ax3.set_title('Distribuição de Tipos de Dados', fontweight='bold')

        # 4. Correlação (se houver dados numéricos)
        ax4 = fig.add_subplot(gs[1, 1])
        numeric_df = df.select_dtypes(include=[np.number])

        if not numeric_df.empty and len(numeric_df.columns) > 1:
            correlation_matrix = numeric_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax4)
            ax4.set_title('Matriz de Correlação', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Dados insuficientes\npara correlação', ha='center', va='center')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Correlação', fontweight='bold')

        plt.suptitle('Visão Geral do Dataset', fontsize=16, fontweight='bold')

        # Salvar
        filename = f"overview_{session_id}_{datetime.now().strftime('%H%M%S')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def create_summary_dashboard(self, df: pd.DataFrame, session_id: str) -> str:
        """Cria dashboard resumo completo"""
        try:
            return self._create_overview_charts(df, session_id)
        except Exception as e:
            print(f"❌ Erro ao criar dashboard: {e}")
            return None
