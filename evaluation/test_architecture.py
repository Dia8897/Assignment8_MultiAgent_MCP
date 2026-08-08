import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.nodes.supervisor import supervisor
from tools.product_decision_tool import analyze_products


def base_state() -> dict:
    return {
        "message": "test request",
        "conversation_history": "",
        "retrieved_evidence": "",
        "completed_agents": [],
        "required_agents": [],
        "agent_tasks": {},
        "specialist_results": [],
        "iteration_count": 0,
    }


class SupervisorRouteTests(unittest.TestCase):
    def route(self, agents: list[str]) -> dict:
        plan = {
            "required_agents": agents,
            "agent_tasks": {agent: "focused task" for agent in agents},
        }
        class FakeLLM:
            def bind(self, **_kwargs):
                return self

            def invoke(self, _prompt):
                return SimpleNamespace(text=json.dumps(plan))

        fake_llm = FakeLLM()
        with patch("agent.nodes.supervisor.llm", fake_llm):
            return supervisor(base_state())

    def test_general_route(self):
        result = self.route(["general_agent"])
        self.assertEqual(result["next_agent"], "general_agent")

    def test_rag_route(self):
        result = self.route(["retrieval_agent"])
        self.assertEqual(result["next_agent"], "retrieval_agent")

    def test_recall_route(self):
        result = self.route(["recall_agent"])
        self.assertEqual(result["next_agent"], "recall_agent")

    def test_comparison_always_starts_with_retrieval(self):
        result = self.route(["comparison_agent"])
        self.assertEqual(
            result["required_agents"],
            ["retrieval_agent", "comparison_agent"],
        )
        self.assertEqual(result["next_agent"], "retrieval_agent")


class ProductAnalyzerTests(unittest.TestCase):
    def test_missing_unit_sizes_refuses_ranking(self):
        result = analyze_products(
            [
                {"name": "A", "current_price": 2, "currency": "EUR"},
                {"name": "B", "current_price": 3, "currency": "EUR"},
            ],
            {"unit_price": 1, "total_price": 0, "ingredients": 0, "size": 0},
        )
        self.assertEqual(result["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
