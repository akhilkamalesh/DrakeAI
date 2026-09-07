"""Automated integration tests for DrakeAI backend API and LangGraph pipeline."""

import logging
import unittest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.db import check_db_health, search_context_graph
from backend.embeddings import get_embedder
from backend.graph.nodes import OUT_OF_SCOPE_GUIDANCE
from backend.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_backend")


class TestDrakeAIBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_db_health(self):
        """Verify database connectivity if PostgreSQL is running."""
        if not check_db_health():
            self.skipTest("PostgreSQL database is offline or inaccessible in sandbox.")
        self.assertTrue(check_db_health(), "PostgreSQL database should be healthy and responsive.")

    def test_02_health_endpoint(self):
        """Verify GET /health returns expected schema and status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data.get("llm_provider"), "gemini")
        self.assertEqual(data.get("gemini_model"), settings.GEMINI_MODEL)
        self.assertEqual(data.get("gemini_configured"), bool(settings.GEMINI_API_KEY))
        self.assertEqual(data.get("embedding_dim"), 1024)

    def test_03_embedder_dimensions(self):
        """Verify embedder outputs exactly 1024-dimensional normalized vectors."""
        embedder = get_embedder()
        vec = embedder.embed_query("Drake Late Night Drive")
        self.assertEqual(len(vec), 1024, "Embedding vector must be 1024-dimensional.")
        self.assertNotEqual(sum(vec[:384]), 0, "Initial 384 dimensions must contain embeddings.")
        self.assertEqual(sum(vec[384:]), 0, "Remaining dimensions (384-1024) must be zero-padded.")

    def test_04_context_graph_retrieval(self):
        """Verify multi-table JOIN across stanza, track, and album with metadata filters."""
        if not check_db_health():
            self.skipTest("PostgreSQL database is offline or inaccessible in sandbox.")
        embedder = get_embedder()
        vec = embedder.embed_query("heartbreak and late night thoughts")
        results = search_context_graph(
            query_vector=vec,
            filters={"release_year_before": 2018, "limit": 3}
        )
        self.assertGreater(len(results), 0, "Should retrieve at least one stanza.")
        row = results[0]
        self.assertIn("stanza_id", row)
        self.assertIn("track_name", row)
        self.assertIn("album_name", row)
        self.assertIn("release_date", row)
        self.assertIn("lyric_chunk", row)
        self.assertIn("similarity", row)
        self.assertIn("track_lyrics", row, "Parent-child retrieval must include full parent track_lyrics.")
        self.assertIsNotNone(row["track_lyrics"], "track_lyrics should not be None.")
        # Year verification
        if row.get("release_date"):
            year = int(str(row["release_date"])[:4])
            self.assertLess(year, 2018, f"Release year {year} must be before 2018.")

        # Test excluded_track_ids parameter
        first_track_id = row["track_id"]
        excluded_results = search_context_graph(
            query_vector=vec,
            filters={"release_year_before": 2018, "limit": 3},
            excluded_track_ids=[first_track_id]
        )
        if excluded_results:
            excluded_ids = [r["track_id"] for r in excluded_results]
            self.assertNotIn(first_track_id, excluded_ids, "Excluded track ID must not appear in retrieval.")

    def test_05_out_of_scope_guardrail(self):
        """Verify out-of-scope query triggers early exit guidance without retrieval."""
        response = self.client.post(
            "/api/chat",
            json={
                "prompt": "How do I bake a chocolate cake?",
                "history": []
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertEqual(data["response"], OUT_OF_SCOPE_GUIDANCE)
        self.assertEqual(len(data.get("sources", [])), 0)

    def test_06_in_scope_chat(self):
        """Verify in-scope query executes full pipeline and returns structured response."""
        if not check_db_health():
            self.skipTest("PostgreSQL database is offline or inaccessible in sandbox.")
        response = self.client.post(
            "/api/chat",
            json={
                "prompt": "Show me introspective Drake lyrics from More Life.",
                "history": []
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertGreater(len(data["response"]), 20)
        self.assertIn("sources", data)
        self.assertGreater(len(data["sources"]), 0, "Should return at least one source track.")
        first_src = data["sources"][0]
        self.assertIn("track_name", first_src)
        self.assertIn("album_name", first_src)
        self.assertIn("quoted_stanzas", first_src)
        self.assertIsNotNone(first_src.get("match_rationale"), "Source must contain match rationale from reasoning agent.")
        self.assertGreater(len(first_src["match_rationale"]), 10)

    def test_07_heuristic_reasoning(self):
        """Verify heuristic reasoning generates analytical thematic analysis and match rationales."""
        from backend.graph.nodes import _heuristic_reasoning
        mock_context = [
            {
                "track_id": "track_1",
                "track_name": "Passionfruit",
                "album_name": "More Life",
                "release_date": "2017-03-18",
                "personal_feel": "Late-Night Confessional",
                "lyric_chunk": "Listen, seeing you got, seeing you got me started"
            }
        ]
        result = _heuristic_reasoning("introspective relationship lyrics", mock_context)
        self.assertIn("thematic_analysis", result)
        self.assertIn("document_rationales", result)
        self.assertIn("Passionfruit", result["document_rationales"])
        self.assertIn("More Life", result["document_rationales"]["Passionfruit"])
        self.assertIn("Late-Night Confessional", result["document_rationales"]["Passionfruit"])

    def test_08_reasoning_node_empty_context(self):
        """Verify reasoning agent handles empty retrieval gracefully."""
        from backend.graph.nodes import reasoning_agent_node
        empty_state = {
            "prompt": "Drake songs",
            "retrieved_context": [],
            "history": []
        }
        res = reasoning_agent_node(empty_state)
        self.assertIsNone(res["reasoning_analysis"])
        self.assertEqual(res["document_rationales"], {})

    def test_09_semantic_query_expansion(self):
        """Verify semantic query expansion extracts and enriches emotional concepts like 'gutwrenching'."""
        from backend.graph.nodes import _heuristic_guardrail_and_intent
        res = _heuristic_guardrail_and_intent("Find me a gutwrenching Drake song about heartbreak")
        self.assertTrue(res.is_relevant)
        self.assertIsNotNone(res.semantic_query)
        expanded = res.semantic_query.lower()
        self.assertIn("heartbreaking", expanded)
        self.assertIn("agonizing", expanded)
        self.assertIn("painful", expanded)

    def test_10_parent_child_vetting_approved(self):
        """Verify vet_track_node approves relevant candidate track with full parent lyrics."""
        from backend.graph.nodes import vet_track_node
        state = {
            "prompt": "Show me a sad late night song",
            "semantic_query": "sad late night song heartbreak crying regret",
            "retrieved_context": [
                {
                    "track_id": "t_marvins",
                    "track_name": "Marvins Room",
                    "album_name": "Take Care",
                    "personal_feel": "Late-Night Confessional",
                    "similarity": 0.82,
                    "lyric_chunk": "Cups of the Rosé, bitches in my old phone",
                    "track_lyrics": "Cups of the Rosé, bitches in my old phone\nI should call her and tell her that I miss her..."
                }
            ],
            "excluded_track_ids": [],
            "retry_count": 0
        }
        result = vet_track_node(state)
        self.assertTrue(result["is_vetted"])
        self.assertEqual(len(result["retrieved_context"]), 1)
        self.assertNotIn("t_marvins", result.get("excluded_track_ids", []))

    def test_11_parent_child_vetting_rejected_and_retry(self):
        """Verify vet_track_node rejects contradictory candidate track and excludes it for retry."""
        from backend.graph.nodes import vet_track_node
        state = {
            "prompt": "Find a gutwrenching song about sorrow and heartbreak",
            "semantic_query": "gutwrenching sorrow heartbreak crying tears agony",
            "retrieved_context": [
                {
                    "track_id": "t_club",
                    "track_name": "Jumpman",
                    "album_name": "What a Time to Be Alive",
                    "personal_feel": "The Club Anthem",
                    "similarity": 0.35,
                    "lyric_chunk": "Jumpman, Jumpman, Jumpman, them boys up to somethin'",
                    "track_lyrics": "Jumpman, Jumpman, Jumpman, them boys up to somethin'\nWoo, just spent the night in the club..."
                }
            ],
            "excluded_track_ids": [],
            "retry_count": 0
        }
        result = vet_track_node(state)
        self.assertFalse(result["is_vetted"])
        self.assertEqual(len(result["retrieved_context"]), 0)
        self.assertIn("t_club", result["excluded_track_ids"])
        self.assertEqual(result["retry_count"], 1)

    def test_12_workflow_routing(self):
        """Verify route_vetting conditionally routes between hybrid_retrieval retry and reasoning_agent."""
        from backend.graph.workflow import route_vetting
        # 1. Approved case
        approved_state = {"is_vetted": True, "retry_count": 0}
        self.assertEqual(route_vetting(approved_state), "reasoning_agent")

        # 2. Rejected retry case (retry_count <= MAX_RETRIES)
        retry_state = {"is_vetted": False, "retry_count": 1}
        self.assertEqual(route_vetting(retry_state), "hybrid_retrieval")

        # 3. Max retries exceeded
        exceeded_state = {"is_vetted": False, "retry_count": 3}
        self.assertEqual(route_vetting(exceeded_state), "reasoning_agent")


if __name__ == "__main__":
    unittest.main()

