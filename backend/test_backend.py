"""Automated integration tests for DrakeAI backend API and LangGraph pipeline."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
        """Verify heuristic reasoning generates candid thematic analysis and match rationales in Drake's persona."""
        from backend.graph.nodes import _heuristic_reasoning
        mock_context = [
            {
                "track_id": "track_1",
                "track_name": "Passionfruit",
                "album_name": "More Life",
                "release_date": "2017-03-18",
                "lyric_chunk": "Listen, seeing you got, seeing you got me started"
            }
        ]
        result = _heuristic_reasoning("introspective relationship lyrics", mock_context)
        self.assertIn("thematic_analysis", result)
        self.assertIn("document_rationales", result)
        self.assertIn("Passionfruit", result["document_rationales"])
        self.assertIn("More Life", result["document_rationales"]["Passionfruit"])
        # Verify Drake first-person voice
        self.assertTrue(any(w in result["document_rationales"]["Passionfruit"].lower() for w in ["i", "me", "honest", "feelings"]))
        self.assertTrue(any(w in result["thematic_analysis"].lower() for w in ["i", "my", "records", "toronto"]))

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
                    "similarity": 0.82,
                    "valence": 0.28,
                    "energy": 0.32,
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
                    "similarity": 0.35,
                    "valence": 0.85,
                    "energy": 0.88,
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

    def test_13_audio_models_serialization(self):
        """Verify AudioFeatureFilters, AudioFeatureTargets, and MetadataFilters serialize and validate cleanly."""
        from backend.models import AudioFeatureFilters, AudioFeatureTargets, MetadataFilters
        af = AudioFeatureFilters(max_valence=0.35, max_energy=0.50, mode=0)
        at = AudioFeatureTargets(target_valence=0.20, target_energy=0.30, target_tempo=80.0)
        mf = MetadataFilters(limit=3, audio_filters=af, audio_targets=at, sort_by="valence_asc")
        d = mf.model_dump()
        self.assertEqual(d["audio_filters"]["max_valence"], 0.35)
        self.assertEqual(d["audio_filters"]["mode"], 0)
        self.assertEqual(d["audio_targets"]["target_tempo"], 80.0)
        self.assertEqual(d["sort_by"], "valence_asc")

    def test_14_heuristic_audio_intent_extraction(self):
        """Verify heuristic intent extraction extracts audio filters and targets for sad and hype queries."""
        from backend.graph.nodes import _heuristic_guardrail_and_intent
        # 1. Saddest query
        res_sad = _heuristic_guardrail_and_intent("What are the top 3 saddest Drake songs?")
        self.assertTrue(res_sad.is_relevant)
        self.assertEqual(res_sad.metadata_filters.limit, 3)
        self.assertIsNotNone(res_sad.metadata_filters.audio_filters)
        self.assertLessEqual(res_sad.metadata_filters.audio_filters.max_valence, 0.40)
        self.assertIsNotNone(res_sad.metadata_filters.audio_targets)
        self.assertEqual(res_sad.metadata_filters.audio_targets.target_valence, 0.20)
        self.assertEqual(res_sad.metadata_filters.sort_by, "valence_asc")

        # 2. Hype query
        res_hype = _heuristic_guardrail_and_intent("Give me 5 high energy club bangers")
        self.assertTrue(res_hype.is_relevant)
        self.assertEqual(res_hype.metadata_filters.limit, 5)
        self.assertGreaterEqual(res_hype.metadata_filters.audio_filters.min_energy, 0.65)
        self.assertEqual(res_hype.metadata_filters.sort_by, "energy_desc")

        # 3. Explicit numeric constraints
        res_num = _heuristic_guardrail_and_intent("Drake songs with valence < 0.25 and tempo > 130")
        self.assertTrue(res_num.is_relevant)
        self.assertEqual(res_num.metadata_filters.audio_filters.max_valence, 0.25)
        self.assertEqual(res_num.metadata_filters.audio_filters.min_tempo, 130.0)

    def test_15_audio_similarity_computation(self):
        """Verify _compute_audio_similarity computes weighted distance between track and target profile."""
        from backend.db import _compute_audio_similarity
        target = {
            "target_valence": 0.20,
            "target_energy": 0.30,
            "target_tempo": 80.0
        }
        # Close match
        close_track = {"valence": 0.22, "energy": 0.32, "tempo": 82.0}
        sim_close = _compute_audio_similarity(close_track, target)
        self.assertGreater(sim_close, 0.90)

        # Distant match (high energy party track)
        distant_track = {"valence": 0.85, "energy": 0.90, "tempo": 130.0}
        sim_distant = _compute_audio_similarity(distant_track, target)
        self.assertLess(sim_distant, 0.50)
        self.assertGreater(sim_close, sim_distant)

    def test_16_vet_track_audio_feature_mismatch(self):
        """Verify _heuristic_vet_track flags track if musical valence directly contradicts sad intent."""
        from backend.graph.nodes import _heuristic_vet_track
        happy_track = {
            "track_name": "In My Feelings",
            "personal_feel": "The Club Anthem",
            "valence": 0.84,
            "energy": 0.62,
            "similarity": 0.55,
            "track_lyrics": "Kiki, do you love me? Are you riding?",
            "lyric_chunk": "Kiki, do you love me?"
        }
        res = _heuristic_vet_track("Give me a gutwrenching Drake song about heartbreak", "sad heartbreak crying tears", happy_track)
        self.assertFalse(res.is_match)
        self.assertIn("valence", res.reason.lower())

    def test_17_source_item_audio_features(self):
        """Verify SourceItem parses and stores audio features."""
        from backend.models import SourceItem
        src = SourceItem(
            track_name="Marvins Room",
            album_name="Take Care",
            release_date="2011-11-15",
            valence=0.31,
            energy=0.34,
            tempo=86.0,
            danceability=0.55,
            similarity=0.88,
            hybrid_score=0.91
        )
        self.assertEqual(src.valence, 0.31)
        self.assertEqual(src.tempo, 86.0)
        self.assertEqual(src.hybrid_score, 0.91)

    def test_18_search_context_graph_hybrid_scoring_and_relaxation(self):
        """Verify search_context_graph multi-modal hybrid scoring and sorting with mocked DB."""
        from unittest.mock import MagicMock, patch
        from backend.db import search_context_graph

        sample_rows = [
            {
                "stanza_id": 1,
                "track_id": "t1",
                "track_name": "Marvins Room",
                "album_name": "Take Care",
                "release_date": "2011-11-15",
                "similarity": 0.85,
                "valence": 0.25,
                "energy": 0.35,
                "tempo": 80.0,
                "personal_feel": "Late-Night Confessional",
                "lyric_chunk": "Cups of the Rosé...",
                "track_lyrics": "Cups of the Rosé..."
            },
            {
                "stanza_id": 2,
                "track_id": "t2",
                "track_name": "In My Feelings",
                "album_name": "Scorpion",
                "release_date": "2018-06-29",
                "similarity": 0.88,
                "valence": 0.84,
                "energy": 0.65,
                "tempo": 91.0,
                "personal_feel": "The Club Anthem",
                "lyric_chunk": "Kiki do you love me...",
                "track_lyrics": "Kiki do you love me..."
            }
        ]

        with patch("backend.db.get_db_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cur
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            mock_cur.fetchall.return_value = sample_rows

            query_vec = [0.0] * 1024
            filters = {
                "limit": 2,
                "audio_targets": {"target_valence": 0.20, "target_energy": 0.30},
                "sort_by": "valence_asc"
            }
            results = search_context_graph(query_vec, filters)
            self.assertEqual(len(results), 2)
            # With valence_asc, Marvins Room (0.25) must be ranked before In My Feelings (0.84)
            self.assertEqual(results[0]["track_name"], "Marvins Room")
            self.assertEqual(results[1]["track_name"], "In My Feelings")
            self.assertIn("hybrid_score", results[0])
            self.assertIn("valence", results[0])
            self.assertGreater(results[0]["hybrid_score"], 0.70)

    def test_19_keyword_extraction(self):
        """Verify _extract_query_keywords extracts emotional themes, categories, and albums."""
        from backend.graph.nodes import _extract_query_keywords
        kws = _extract_query_keywords("What are the top 3 saddest Drake songs about heartbreak on Take Care?")
        self.assertIn("saddest", kws)
        self.assertIn("heartbreak", kws)
        self.assertIn("Album: Take Care", kws)

    def test_20_pulled_tracks_and_vetting_decisions(self):
        """Verify hybrid_retrieval_node formats pulled_tracks and vet_track_node accumulates vetting_decisions."""
        from backend.graph.nodes import hybrid_retrieval_node, vet_track_node
        from unittest.mock import patch

        mock_rows = [
            {
                "stanza_id": 10,
                "track_id": "tr_1",
                "track_name": "Doing It Wrong",
                "album_name": "Take Care",
                "release_date": "2011-11-15",
                "personal_feel": "Late-Night Confessional",
                "similarity": 0.82,
                "hybrid_score": 0.85,
                "valence": 0.28,
                "energy": 0.30,
                "tempo": 84.0,
                "danceability": 0.45,
                "lyric_chunk": "When a good thing goes bad, it's not the end of the world",
                "track_lyrics": "When a good thing goes bad, it's not the end of the world..."
            }
        ]

        with patch("backend.graph.nodes.search_context_graph", return_value=mock_rows):
            state = {
                "prompt": "saddest Drake song",
                "semantic_query": "sad heartbreak melancholy",
                "metadata_filters": {},
                "excluded_track_ids": []
            }
            res_retrieval = hybrid_retrieval_node(state)
            self.assertIn("pulled_tracks", res_retrieval)
            self.assertEqual(len(res_retrieval["pulled_tracks"]), 1)
            first_pulled = res_retrieval["pulled_tracks"][0]
            self.assertEqual(first_pulled["track_name"], "Doing It Wrong")
            self.assertEqual(first_pulled["valence"], 0.28)

            # Test vet_track_node
            state_to_vet = {
                "prompt": "saddest Drake song",
                "semantic_query": "sad heartbreak melancholy",
                "retrieved_context": mock_rows,
                "excluded_track_ids": [],
                "retry_count": 0,
                "vetting_decisions": []
            }
            res_vet = vet_track_node(state_to_vet)
            self.assertTrue(res_vet["is_vetted"])
            self.assertIn("vetting_decisions", res_vet)
            self.assertEqual(len(res_vet["vetting_decisions"]), 1)
            dec = res_vet["vetting_decisions"][0]
            self.assertEqual(dec["track_name"], "Doing It Wrong")
            self.assertEqual(dec["status"], "APPROVED")
            self.assertTrue(dec["is_match"])
            self.assertGreater(dec["confidence"], 0.5)

    def test_21_agent_trace_compilation(self):
        """Verify response_formatter_node compiles unified agent_trace structure."""
        from backend.graph.nodes import response_formatter_node

        mock_context = [
            {
                "track_id": "tr_1",
                "track_name": "Doing It Wrong",
                "album_name": "Take Care",
                "release_date": "2011-11-15",
                "personal_feel": "Late-Night Confessional",
                "similarity": 0.82,
                "valence": 0.28,
                "energy": 0.30,
                "tempo": 84.0,
                "lyric_chunk": "When a good thing goes bad"
            }
        ]
        mock_pulled = [
            {
                "track_name": "Doing It Wrong",
                "album_name": "Take Care",
                "similarity": 0.82,
                "valence": 0.28
            }
        ]
        mock_vetting = [
            {
                "track_name": "Doing It Wrong",
                "status": "APPROVED",
                "is_match": True,
                "confidence": 0.9,
                "reason": "Low acoustic valence matches melancholy query."
            }
        ]

        state = {
            "prompt": "saddest Drake song",
            "extracted_keywords": ["saddest", "Album: Take Care"],
            "semantic_query": "sad melancholy",
            "metadata_filters": {"limit": 1, "sort_by": "valence_asc"},
            "retrieved_context": mock_context,
            "pulled_tracks": mock_pulled,
            "is_vetted": True,
            "retry_count": 0,
            "vetting_rationale": "Matches heartbreak criteria",
            "vetting_decisions": mock_vetting,
            "reasoning_analysis": "Doing It Wrong captures Drake at his most vulnerable.",
            "document_rationales": {"Doing It Wrong": "Direct match for heartbreak themes."}
        }

        res = response_formatter_node(state)
        self.assertIn("agent_trace", res)
        trace = res["agent_trace"]
        self.assertIn("query_analysis", trace)
        self.assertEqual(trace["query_analysis"]["keywords"], ["saddest", "Album: Take Care"])
        self.assertIn("pulled_songs", trace)
        self.assertEqual(len(trace["pulled_songs"]), 1)
        self.assertIn("vetting_agent", trace)
        self.assertTrue(trace["vetting_agent"]["is_vetted"])
        self.assertEqual(len(trace["vetting_agent"]["decisions"]), 1)

    def test_22_chat_response_agent_trace_api(self):
        """Verify POST /api/chat includes agent_trace in response."""
        response = self.client.post(
            "/api/chat",
            json={
                "prompt": "How do I bake bread?",
                "history": []
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("agent_trace", data)
        self.assertIsNotNone(data["agent_trace"])
        self.assertIn("query_analysis", data["agent_trace"])

    def test_23_chat_stream_endpoint(self):
        """Verify POST /api/chat/stream yields newline-delimited JSON events."""
        import json
        response = self.client.post(
            "/api/chat/stream",
            json={
                "prompt": "What is the weather today?",
                "history": []
            }
        )
        self.assertEqual(response.status_code, 200)
        lines = [line for line in response.text.split("\n") if line.strip()]
        self.assertGreater(len(lines), 0)
        # Parse each line as JSON
        events = [json.loads(line) for line in lines]
        steps = [e.get("step") for e in events]
        self.assertIn("query_analysis", steps)
        self.assertIn("complete", steps)

    def test_24_drake_persona_and_vibe_deprecation(self):
        """Verify Drake persona tone and deprecation of vibe categories across reasoning and formatter."""
        from backend.graph.nodes import _heuristic_guardrail_and_intent, _heuristic_reasoning, response_formatter_node
        # 1. Guardrail does not set personal_feel
        intent = _heuristic_guardrail_and_intent("Find late-night confessional songs")
        self.assertIsNone(intent.metadata_filters.personal_feel)

        # 2. Reasoning outputs Drake first-person reflection
        mock_ctx = [
            {
                "track_id": "tr_marvins",
                "track_name": "Marvins Room",
                "album_name": "Take Care",
                "release_date": "2011-11-15",
                "lyric_chunk": "I'm just sayin' you could do better",
                "valence": 0.28,
                "tempo": 86.0
            }
        ]
        reasoning = _heuristic_reasoning("late night thoughts", mock_ctx)
        drake_rationale = reasoning["document_rationales"]["Marvins Room"]
        # Confirm authentic first-person Drake phrasing
        self.assertTrue(any(phrase in drake_rationale.lower() for phrase in ["honest moment for me", "my real feelings", "man, 'marvins room'"]))
        self.assertNotIn("under the '", drake_rationale)
        self.assertNotIn("category", drake_rationale.lower())

        # 3. Response formatter produces Drake's reflection without vibe badges
        state = {
            "prompt": "late night thoughts",
            "extracted_keywords": ["late night"],
            "semantic_query": "late night thoughts",
            "metadata_filters": {"limit": 1},
            "retrieved_context": mock_ctx,
            "pulled_tracks": [],
            "is_vetted": True,
            "retry_count": 0,
            "vetting_rationale": "Vetted",
            "vetting_decisions": [],
            "reasoning_analysis": reasoning["thematic_analysis"],
            "document_rationales": reasoning["document_rationales"]
        }
        res = response_formatter_node(state)
        final_resp = res["final_response"]
        self.assertIn("Drake's Reflection", final_resp)
        self.assertNotIn("Vibe / Category", final_resp)
        self.assertIsNone(res["agent_trace"]["query_analysis"]["metadata_limits"]["personal_feel"])


