
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os

class ConversationMemory:
    """
    Sistema de memória para armazenar e recuperar interações do agente EDA
    """

    def __init__(self, max_interactions: int = 50):
        """
        Inicializa o sistema de memória

        Args:
            max_interactions: Número máximo de interações a manter na memória
        """
        self.max_interactions = max_interactions
        self.memory: List[Dict[str, Any]] = []
        self.current_session = None

        # Criar diretório para persistência
        self.memory_dir = "outputs/memory"
        os.makedirs(self.memory_dir, exist_ok=True)

    def save_interaction(self, session_id: str, question: str, result: Dict[str, Any]) -> None:
        """
        Salva uma interação na memória

        Args:
            session_id: ID da sessão
            question: Pergunta feita pelo usuário
            result: Resultado da análise
        """
        interaction = {
            "session_id": session_id,
            "question": question,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "analysis_type": self._classify_question(question)
        }

        # Adicionar à memória
        self.memory.append(interaction)

        # Manter apenas as últimas interações
        if len(self.memory) > self.max_interactions:
            self.memory = self.memory[-self.max_interactions:]

        print(f"💾 Interação salva na memória: {question[:50]}...")

    def get_recent_interactions(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera as interações mais recentes

        Args:
            count: Número de interações a retornar

        Returns:
            Lista das interações mais recentes
        """
        return self.memory[-count:] if self.memory else []

    def get_session_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Recupera todas as interações de uma sessão específica

        Args:
            session_id: ID da sessão

        Returns:
            Lista de interações da sessão
        """
        return [
            interaction for interaction in self.memory 
            if interaction.get("session_id") == session_id
        ]

    def get_similar_questions(self, question: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Encontra perguntas similares feitas anteriormente

        Args:
            question: Pergunta atual
            limit: Número máximo de perguntas similares

        Returns:
            Lista de interações com perguntas similares
        """
        question_lower = question.lower()
        similar_interactions = []

        # Palavras-chave para similaridade
        question_keywords = set(question_lower.split())

        for interaction in self.memory:
            prev_question = interaction["question"].lower()
            prev_keywords = set(prev_question.split())

            # Calcular similaridade simples baseada em palavras comuns
            common_keywords = question_keywords.intersection(prev_keywords)
            similarity = len(common_keywords) / max(len(question_keywords), len(prev_keywords))

            if similarity > 0.3:  # Threshold de similaridade
                interaction_copy = interaction.copy()
                interaction_copy["similarity"] = similarity
                similar_interactions.append(interaction_copy)

        # Ordenar por similaridade
        similar_interactions.sort(key=lambda x: x["similarity"], reverse=True)

        return similar_interactions[:limit]

    def _classify_question(self, question: str) -> str:
        """
        Classifica o tipo de pergunta para organizar a memória

        Args:
            question: Pergunta a ser classificada

        Returns:
            Tipo da pergunta
        """
        question_lower = question.lower()

        # Mapeamento de palavras-chave para tipos
        classifications = {
            "correlação": ["correlação", "correlacion", "correlation", "relacionamento"],
            "distribuição": ["distribuição", "distribution", "histograma", "histogram"],
            "outliers": ["outlier", "outliers", "anomalia", "anomalias", "atípico"],
            "missing": ["ausente", "missing", "faltante", "nulo", "null", "nan"],
            "estatísticas": ["média", "mean", "mediana", "median", "desvio", "std", "estatística"],
            "visualização": ["gráfico", "chart", "plot", "visualização", "visualization"],
            "geral": ["resumo", "summary", "geral", "overview", "análise geral"]
        }

        for category, keywords in classifications.items():
            if any(keyword in question_lower for keyword in keywords):
                return category

        return "outros"

    def get_context_summary(self, session_id: str) -> str:
        """
        Gera resumo do contexto da sessão atual

        Args:
            session_id: ID da sessão

        Returns:
            Resumo textual do contexto
        """
        session_interactions = self.get_session_interactions(session_id)

        if not session_interactions:
            return "Nenhuma interação anterior na sessão."

        summary_parts = [
            f"CONTEXTO DA SESSÃO ({len(session_interactions)} interações):",
            ""
        ]

        # Agrupar por tipo de análise
        analysis_types = {}
        for interaction in session_interactions:
            analysis_type = interaction.get("analysis_type", "outros")
            if analysis_type not in analysis_types:
                analysis_types[analysis_type] = []
            analysis_types[analysis_type].append(interaction)

        # Resumir cada tipo
        for analysis_type, interactions in analysis_types.items():
            summary_parts.append(f"📊 {analysis_type.upper()}: {len(interactions)} análise(s)")
            for interaction in interactions[-2:]:  # Últimas 2 de cada tipo
                summary_parts.append(f"   - {interaction['question'][:60]}...")
            summary_parts.append("")

        return "\n".join(summary_parts)

    def save_session_to_file(self, session_id: str) -> str:
        """
        Salva sessão completa em arquivo JSON

        Args:
            session_id: ID da sessão

        Returns:
            Caminho do arquivo salvo
        """
        try:
            session_interactions = self.get_session_interactions(session_id)

            session_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "total_interactions": len(session_interactions),
                "interactions": session_interactions
            }

            filename = f"session_{session_id}.json"
            filepath = os.path.join(self.memory_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Sessão salva em: {filepath}")
            return filepath

        except Exception as e:
            print(f"❌ Erro ao salvar sessão: {e}")
            return ""

    def load_session_from_file(self, session_id: str) -> bool:
        """
        Carrega sessão de arquivo JSON

        Args:
            session_id: ID da sessão

        Returns:
            True se carregou com sucesso
        """
        try:
            filename = f"session_{session_id}.json"
            filepath = os.path.join(self.memory_dir, filename)

            if not os.path.exists(filepath):
                print(f"⚠️ Arquivo de sessão não encontrado: {filepath}")
                return False

            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            # Adicionar interações à memória atual
            for interaction in session_data.get("interactions", []):
                self.memory.append(interaction)

            # Manter limite da memória
            if len(self.memory) > self.max_interactions:
                self.memory = self.memory[-self.max_interactions:]

            print(f"✅ Sessão carregada: {len(session_data.get('interactions', []))} interações")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar sessão: {e}")
            return False

    def clear_session(self):
        """Limpa a memória da sessão atual"""
        self.memory = []
        self.current_session = None
        print("🔄 Memória da sessão limpa")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas da memória

        Returns:
            Estatísticas da memória
        """
        if not self.memory:
            return {"total_interactions": 0}

        # Análise por tipo
        analysis_types = {}
        sessions = set()

        for interaction in self.memory:
            analysis_type = interaction.get("analysis_type", "outros")
            analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1

            session_id = interaction.get("session_id")
            if session_id:
                sessions.add(session_id)

        # Interação mais recente
        latest_interaction = self.memory[-1]["timestamp"] if self.memory else None

        return {
            "total_interactions": len(self.memory),
            "unique_sessions": len(sessions),
            "analysis_types": analysis_types,
            "latest_interaction": latest_interaction,
            "memory_usage_percent": (len(self.memory) / self.max_interactions) * 100
        }

    def export_memory_summary(self) -> str:
        """
        Exporta resumo completo da memória

        Returns:
            Texto com resumo da memória
        """
        stats = self.get_memory_stats()

        summary = f"""
📊 RESUMO DA MEMÓRIA DO AGENTE EDA

Total de Interações: {stats['total_interactions']}
Sessões Únicas: {stats['unique_sessions']}
Uso da Memória: {stats['memory_usage_percent']:.1f}%

📈 TIPOS DE ANÁLISE:
"""

        for analysis_type, count in stats.get('analysis_types', {}).items():
            summary += f"  • {analysis_type}: {count} análises\n"

        if stats.get('latest_interaction'):
            summary += f"\n🕒 Última Interação: {stats['latest_interaction']}"

        return summary
