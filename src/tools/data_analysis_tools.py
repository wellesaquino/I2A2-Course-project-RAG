
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataAnalysisTools:
    """
    Ferramentas especializadas para análise exploratória de dados
    """

    def __init__(self):
        self.analysis_cache = {}

    def basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extrai informações básicas do dataframe

        Args:
            df: DataFrame para análise

        Returns:
            Dicionário com informações básicas
        """
        try:
            info = {
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                "dtypes": df.dtypes.value_counts().to_dict(),
                "column_info": {}
            }

            # Informações por coluna
            for col in df.columns:
                col_info = {
                    "dtype": str(df[col].dtype),
                    "non_null_count": df[col].count(),
                    "null_count": df[col].isnull().sum(),
                    "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2),
                    "unique_count": df[col].nunique(),
                    "unique_percentage": round((df[col].nunique() / len(df)) * 100, 2)
                }

                # Informações específicas por tipo
                if df[col].dtype in ['int64', 'float64']:
                    col_info.update({
                        "mean": round(df[col].mean(), 4),
                        "median": round(df[col].median(), 4),
                        "std": round(df[col].std(), 4),
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "q25": round(df[col].quantile(0.25), 4),
                        "q75": round(df[col].quantile(0.75), 4)
                    })

                info["column_info"][col] = col_info

            return info

        except Exception as e:
            return {"error": f"Erro ao extrair informações básicas: {e}"}

    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta e analisa valores ausentes

        Args:
            df: DataFrame para análise

        Returns:
            Análise completa de valores ausentes
        """
        try:
            missing_data = df.isnull().sum()
            missing_percentage = (missing_data / len(df)) * 100

            result = {
                "total_missing": missing_data.sum(),
                "missing_percentage_total": round((missing_data.sum() / df.size) * 100, 2),
                "columns_with_missing": [],
                "missing_patterns": {}
            }

            # Colunas com valores ausentes
            for col in missing_data[missing_data > 0].index:
                result["columns_with_missing"].append({
                    "column": col,
                    "missing_count": int(missing_data[col]),
                    "missing_percentage": round(missing_percentage[col], 2)
                })

            # Padrões de valores ausentes
            if len(result["columns_with_missing"]) > 0:
                missing_matrix = df.isnull()
                patterns = missing_matrix.value_counts()

                for pattern, count in patterns.head(5).items():
                    pattern_desc = []
                    for i, col in enumerate(df.columns):
                        if pattern[i]:
                            pattern_desc.append(f"{col}: Missing")
                        else:
                            pattern_desc.append(f"{col}: Present")

                    result["missing_patterns"][str(pattern)] = {
                        "count": int(count),
                        "percentage": round((count / len(df)) * 100, 2),
                        "description": pattern_desc
                    }

            return result

        except Exception as e:
            return {"error": f"Erro ao detectar valores ausentes: {e}"}

    def detect_outliers(self, df: pd.DataFrame, method: str = "iqr") -> Dict[str, Any]:
        """
        Detecta outliers em colunas numéricas

        Args:
            df: DataFrame para análise
            method: Método de detecção ('iqr', 'zscore')

        Returns:
            Análise de outliers
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers_info = {}

            for col in numeric_cols:
                if method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

                elif method == "zscore":
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = df[z_scores > 3]

                outliers_info[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": round((len(outliers) / len(df)) * 100, 2),
                    "outlier_values": outliers[col].tolist()[:10] if len(outliers) <= 10 else outliers[col].tolist()[:10] + ["..."],
                    "method": method
                }

                if method == "iqr":
                    outliers_info[col].update({
                        "lower_bound": round(lower_bound, 4),
                        "upper_bound": round(upper_bound, 4),
                        "Q1": round(Q1, 4),
                        "Q3": round(Q3, 4),
                        "IQR": round(IQR, 4)
                    })

            return {
                "method": method,
                "numeric_columns_analyzed": len(numeric_cols),
                "outliers_by_column": outliers_info,
                "total_outliers": sum([info["outlier_count"] for info in outliers_info.values()])
            }

        except Exception as e:
            return {"error": f"Erro ao detectar outliers: {e}"}

    def correlation_analysis(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analisa correlações entre variáveis numéricas

        Args:
            df: DataFrame para análise
            threshold: Limite para correlações significativas

        Returns:
            Análise de correlações
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                return {"error": "Nenhuma coluna numérica encontrada para análise de correlação"}

            correlation_matrix = numeric_df.corr()

            # Encontrar correlações altas
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]

                    if abs(corr_value) >= threshold:
                        high_correlations.append({
                            "variable_1": col1,
                            "variable_2": col2,
                            "correlation": round(corr_value, 4),
                            "abs_correlation": round(abs(corr_value), 4),
                            "strength": self._correlation_strength(abs(corr_value))
                        })

            # Ordenar por correlação absoluta
            high_correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)

            return {
                "correlation_matrix": correlation_matrix.round(4).to_dict(),
                "high_correlations": high_correlations,
                "threshold": threshold,
                "numeric_variables": list(numeric_df.columns),
                "strongest_correlation": high_correlations[0] if high_correlations else None,
                "total_high_correlations": len(high_correlations)
            }

        except Exception as e:
            return {"error": f"Erro na análise de correlação: {e}"}

    def _correlation_strength(self, corr_value: float) -> str:
        """Classifica a força da correlação"""
        if corr_value >= 0.8:
            return "Muito forte"
        elif corr_value >= 0.6:
            return "Forte"
        elif corr_value >= 0.4:
            return "Moderada"
        elif corr_value >= 0.2:
            return "Fraca"
        else:
            return "Muito fraca"

    def categorical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa variáveis categóricas

        Args:
            df: DataFrame para análise

        Returns:
            Análise de variáveis categóricas
        """
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns

            if len(categorical_cols) == 0:
                return {"error": "Nenhuma variável categórica encontrada"}

            categorical_info = {}

            for col in categorical_cols:
                value_counts = df[col].value_counts()

                categorical_info[col] = {
                    "unique_values": df[col].nunique(),
                    "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "most_frequent_percentage": round((value_counts.iloc[0] / len(df)) * 100, 2) if len(value_counts) > 0 else 0,
                    "value_distribution": value_counts.head(10).to_dict(),
                    "is_high_cardinality": df[col].nunique() > len(df) * 0.5
                }

            return {
                "categorical_columns": list(categorical_cols),
                "categorical_analysis": categorical_info,
                "total_categorical_columns": len(categorical_cols)
            }

        except Exception as e:
            return {"error": f"Erro na análise categórica: {e}"}

    def data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera relatório completo de qualidade dos dados

        Args:
            df: DataFrame para análise

        Returns:
            Relatório de qualidade
        """
        try:
            # Análises individuais
            basic = self.basic_info(df)
            missing = self.detect_missing_values(df)
            outliers = self.detect_outliers(df)
            correlations = self.correlation_analysis(df)
            categorical = self.categorical_analysis(df)

            # Pontuação de qualidade
            quality_score = self._calculate_quality_score(basic, missing, outliers)

            return {
                "dataset_overview": basic,
                "missing_values": missing,
                "outliers": outliers,
                "correlations": correlations,
                "categorical_analysis": categorical,
                "quality_score": quality_score,
                "recommendations": self._generate_recommendations(basic, missing, outliers, correlations)
            }

        except Exception as e:
            return {"error": f"Erro ao gerar relatório de qualidade: {e}"}

    def _calculate_quality_score(self, basic: Dict, missing: Dict, outliers: Dict) -> Dict[str, Any]:
        """Calcula pontuação de qualidade dos dados"""
        try:
            score = 100

            # Penalizar por valores ausentes
            missing_penalty = missing.get("missing_percentage_total", 0) * 2
            score -= missing_penalty

            # Penalizar por muitos outliers
            total_outliers = outliers.get("total_outliers", 0)
            total_rows = basic.get("shape", {}).get("rows", 1)
            outlier_percentage = (total_outliers / total_rows) * 100
            outlier_penalty = min(outlier_percentage * 0.5, 20)
            score -= outlier_penalty

            score = max(score, 0)

            return {
                "overall_score": round(score, 1),
                "classification": self._classify_quality(score),
                "missing_penalty": round(missing_penalty, 1),
                "outlier_penalty": round(outlier_penalty, 1)
            }

        except Exception as e:
            return {"overall_score": 0, "error": str(e)}

    def _classify_quality(self, score: float) -> str:
        """Classifica a qualidade dos dados"""
        if score >= 90:
            return "Excelente"
        elif score >= 75:
            return "Boa"
        elif score >= 60:
            return "Regular"
        elif score >= 40:
            return "Ruim"
        else:
            return "Muito ruim"

    def _generate_recommendations(self, basic: Dict, missing: Dict, outliers: Dict, correlations: Dict) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []

        # Recomendações sobre valores ausentes
        if missing.get("missing_percentage_total", 0) > 5:
            recommendations.append("Considere tratar valores ausentes com imputação ou remoção de registros.")

        # Recomendações sobre outliers
        total_outliers = outliers.get("total_outliers", 0)
        total_rows = basic.get("shape", {}).get("rows", 1)
        if (total_outliers / total_rows) > 0.05:
            recommendations.append("Investigate outliers - podem ser erros de entrada ou casos especiais importantes.")

        # Recomendações sobre correlações
        high_corrs = correlations.get("total_high_correlations", 0)
        if high_corrs > 0:
            recommendations.append("Considere feature engineering baseado nas correlações identificadas.")

        # Recomendações gerais
        if basic.get("shape", {}).get("columns", 0) > 20:
            recommendations.append("Dataset com muitas colunas - considere seleção de features.")

        return recommendations
