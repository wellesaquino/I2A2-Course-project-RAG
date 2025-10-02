
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Processador especializado para limpeza e preparação de dados CSV
    """

    def __init__(self):
        self.original_shape = None
        self.processed_shape = None
        self.processing_log = []

    def extract_dataframe_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extrai informações detalhadas do DataFrame

        Args:
            df: DataFrame para análise

        Returns:
            Dicionário com informações completas
        """
        try:
            # Informações básicas
            info = {
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict()
            }

            # Classificar colunas por tipo
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()

            info.update({
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "datetime_columns": datetime_columns,
                "column_count_by_type": {
                    "numeric": len(numeric_columns),
                    "categorical": len(categorical_columns),
                    "datetime": len(datetime_columns)
                }
            })

            # Análise de valores ausentes
            missing_info = self._analyze_missing_values(df)
            info["missing_values_summary"] = missing_info

            # Estatísticas básicas para colunas numéricas
            if numeric_columns:
                numeric_stats = df[numeric_columns].describe().to_dict()
                info["numeric_statistics"] = numeric_stats

            # Análise de cardinalidade para categóricas
            if categorical_columns:
                categorical_info = {}
                for col in categorical_columns:
                    categorical_info[col] = {
                        "unique_count": df[col].nunique(),
                        "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                        "cardinality_ratio": df[col].nunique() / len(df)
                    }
                info["categorical_analysis"] = categorical_info

            # Identificar possíveis problemas
            info["data_quality_issues"] = self._identify_quality_issues(df)

            return info

        except Exception as e:
            return {"error": f"Erro ao extrair informações: {e}"}

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa padrões de valores ausentes"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        return {
            "total_missing_values": int(missing_counts.sum()),
            "columns_with_missing": missing_counts[missing_counts > 0].to_dict(),
            "missing_percentages": missing_percentages[missing_percentages > 0].to_dict(),
            "columns_with_high_missing": missing_percentages[missing_percentages > 50].index.tolist()
        }

    def _identify_quality_issues(self, df: pd.DataFrame) -> List[str]:
        """Identifica possíveis problemas de qualidade nos dados"""
        issues = []

        # Verificar valores ausentes
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 10:
            issues.append(f"Alto percentual de valores ausentes: {missing_percentage:.1f}%")

        # Verificar colunas com única valor
        constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_columns:
            issues.append(f"Colunas com valores constantes: {constant_columns}")

        # Verificar colunas com alta cardinalidade
        high_cardinality_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.8:
                high_cardinality_cols.append(col)

        if high_cardinality_cols:
            issues.append(f"Colunas com alta cardinalidade: {high_cardinality_cols}")

        # Verificar possíveis outliers extremos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        extreme_outlier_cols = []

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                outliers_percentage = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum() / len(df) * 100
                if outliers_percentage > 10:
                    extreme_outlier_cols.append(f"{col} ({outliers_percentage:.1f}%)")

        if extreme_outlier_cols:
            issues.append(f"Colunas com muitos outliers extremos: {extreme_outlier_cols}")

        return issues

    def clean_dataframe(self, df: pd.DataFrame, 
                       remove_duplicates: bool = True,
                       handle_missing: str = "keep",
                       standardize_columns: bool = True) -> pd.DataFrame:
        """
        Limpa e prepara o DataFrame para análise

        Args:
            df: DataFrame original
            remove_duplicates: Se deve remover duplicatas
            handle_missing: Como tratar valores ausentes ('keep', 'drop', 'fill')
            standardize_columns: Se deve padronizar nomes das colunas

        Returns:
            DataFrame limpo
        """
        df_clean = df.copy()
        self.original_shape = df.shape
        self.processing_log = []

        try:
            # Padronizar nomes das colunas
            if standardize_columns:
                original_columns = df_clean.columns.tolist()
                df_clean.columns = self._standardize_column_names(df_clean.columns)
                if list(df_clean.columns) != original_columns:
                    self.processing_log.append("Nomes das colunas padronizados")

            # Remover duplicatas
            if remove_duplicates:
                duplicates_count = df_clean.duplicated().sum()
                if duplicates_count > 0:
                    df_clean = df_clean.drop_duplicates()
                    self.processing_log.append(f"Removidas {duplicates_count} linhas duplicadas")

            # Tratar valores ausentes
            if handle_missing == "drop":
                missing_before = df_clean.isnull().sum().sum()
                df_clean = df_clean.dropna()
                missing_after = df_clean.isnull().sum().sum()
                if missing_before > missing_after:
                    self.processing_log.append(f"Removidas linhas com valores ausentes")

            elif handle_missing == "fill":
                # Preencher valores ausentes de forma inteligente
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                categorical_cols = df_clean.select_dtypes(include=['object']).columns

                for col in numeric_cols:
                    if df_clean[col].isnull().any():
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        self.processing_log.append(f"Valores ausentes em '{col}' preenchidos com mediana")

                for col in categorical_cols:
                    if df_clean[col].isnull().any():
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown')
                        self.processing_log.append(f"Valores ausentes em '{col}' preenchidos com moda")

            # Identificar e corrigir tipos de dados
            df_clean = self._optimize_data_types(df_clean)

            self.processed_shape = df_clean.shape

            return df_clean

        except Exception as e:
            self.processing_log.append(f"Erro durante limpeza: {e}")
            return df

    def _standardize_column_names(self, columns: pd.Index) -> List[str]:
        """Padroniza nomes das colunas"""
        standardized = []

        for col in columns:
            # Converter para string e remover espaços extras
            clean_col = str(col).strip()

            # Substituir espaços por underscore
            clean_col = clean_col.replace(' ', '_')

            # Remover caracteres especiais
            clean_col = ''.join(c for c in clean_col if c.isalnum() or c == '_')

            # Converter para minúsculo
            clean_col = clean_col.lower()

            # Garantir que não comece com número
            if clean_col and clean_col[0].isdigit():
                clean_col = 'col_' + clean_col

            # Garantir que não seja vazio
            if not clean_col:
                clean_col = f'column_{len(standardized)}'

            standardized.append(clean_col)

        return standardized

    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Otimiza tipos de dados para reduzir uso de memória"""
        df_optimized = df.copy()

        # Otimizar colunas numéricas
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            col_data = df_optimized[col]

            # Verificar se pode ser inteiro
            if col_data.dtype == 'float64' and col_data.isnull().sum() == 0:
                if (col_data % 1 == 0).all():
                    # Verificar faixa para escolher tipo de inteiro
                    col_min, col_max = col_data.min(), col_data.max()

                    if col_min >= 0:  # Unsigned integers
                        if col_max < 255:
                            df_optimized[col] = col_data.astype('uint8')
                        elif col_max < 65535:
                            df_optimized[col] = col_data.astype('uint16')
                        elif col_max < 4294967295:
                            df_optimized[col] = col_data.astype('uint32')
                        else:
                            df_optimized[col] = col_data.astype('uint64')
                    else:  # Signed integers
                        if col_min >= -128 and col_max <= 127:
                            df_optimized[col] = col_data.astype('int8')
                        elif col_min >= -32768 and col_max <= 32767:
                            df_optimized[col] = col_data.astype('int16')
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            df_optimized[col] = col_data.astype('int32')
                        else:
                            df_optimized[col] = col_data.astype('int64')

                    self.processing_log.append(f"Coluna '{col}' convertida para inteiro")

            # Converter float64 para float32 quando possível
            elif col_data.dtype == 'float64':
                df_optimized[col] = pd.to_numeric(col_data, downcast='float')

        # Otimizar colunas categóricas
        for col in df_optimized.select_dtypes(include=['object']).columns:
            num_unique_values = df_optimized[col].nunique()
            num_total_values = len(df_optimized[col])

            # Converter para category se baixa cardinalidade
            if num_unique_values / num_total_values < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
                self.processing_log.append(f"Coluna '{col}' convertida para categoria")

        return df_optimized

    def get_processing_summary(self) -> Dict[str, Any]:
        """Retorna resumo do processamento realizado"""
        return {
            "original_shape": self.original_shape,
            "processed_shape": self.processed_shape,
            "processing_log": self.processing_log,
            "memory_reduction": self._calculate_memory_reduction() if self.original_shape and self.processed_shape else None
        }

    def _calculate_memory_reduction(self) -> Dict[str, Any]:
        """Calcula redução de memória (se aplicável)"""
        # Esta função seria mais útil se mantivéssemos referência aos DataFrames
        # Por agora, retorna informação básica
        return {
            "rows_change": self.processed_shape[0] - self.original_shape[0],
            "columns_change": self.processed_shape[1] - self.original_shape[1]
        }

    def validate_csv_file(self, file_path: str) -> Dict[str, Any]:
        """
        Valida arquivo CSV antes do carregamento

        Args:
            file_path: Caminho para o arquivo CSV

        Returns:
            Resultado da validação
        """
        validation_result = {
            "is_valid": False,
            "issues": [],
            "recommendations": []
        }

        try:
            # Verificar se arquivo existe
            if not os.path.exists(file_path):
                validation_result["issues"].append("Arquivo não encontrado")
                return validation_result

            # Verificar tamanho do arquivo
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 100:  # Limite de 100MB
                validation_result["issues"].append(f"Arquivo muito grande: {file_size_mb:.1f}MB")
                validation_result["recommendations"].append("Considere usar uma amostra dos dados")

            # Tentar ler algumas linhas
            try:
                sample_df = pd.read_csv(file_path, nrows=100)
                validation_result["sample_info"] = {
                    "columns": len(sample_df.columns),
                    "sample_rows": len(sample_df),
                    "column_names": sample_df.columns.tolist()[:10]  # Primeiras 10 colunas
                }

                validation_result["is_valid"] = True

            except Exception as e:
                validation_result["issues"].append(f"Erro ao ler CSV: {e}")
                validation_result["recommendations"].append("Verifique se o arquivo está bem formatado")

            return validation_result

        except Exception as e:
            validation_result["issues"].append(f"Erro na validação: {e}")
            return validation_result
