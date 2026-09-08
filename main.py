import json
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="DrakeAI — Lyric & Audio Intelligence",
    page_icon="🎵",
    layout="wide"
)

BACKEND_URL = "http://localhost:8000"

# Custom CSS for rich styling
st.markdown("""
<style>
    .source-card {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 14px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .track-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f3f4f6;
    }
    .track-meta {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 8px;
    }
    .quote-box {
        background-color: rgba(0, 0, 0, 0.25);
        border-left: 3px solid #1DB954;
        padding: 8px 12px;
        border-radius: 4px;
        font-style: italic;
        color: #e5e7eb;
        white-space: pre-wrap;
        margin-top: 6px;
    }
    .stChatMessage img {
        max-width: 180px;
        max-height: 180px;
        border-radius: 8px;
        object-fit: cover;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
        margin-top: 8px;
        margin-bottom: 8px;
        display: block;
    }
    .stChatMessage a[href*="spotify.com"], .spotify-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #1DB954;
        color: white !important;
        padding: 5px 14px;
        border-radius: 20px;
        text-decoration: none;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(29, 185, 84, 0.3);
        transition: all 0.2s ease;
    }
    .stChatMessage a[href*="spotify.com"]:hover, .spotify-btn:hover {
        background-color: #1ed760;
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.5);
        transform: translateY(-1px);
    }
    .response-anchor {
        scroll-margin-top: 80px;
        display: block;
        height: 0;
        width: 0;
    }
    .rationale-box {
        background-color: rgba(29, 185, 84, 0.08);
        border-left: 3px solid #1DB954;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #e5e7eb;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .audio-badge {
        display: inline-block;
        background-color: rgba(29, 185, 84, 0.15);
        border: 1px solid rgba(29, 185, 84, 0.35);
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.78rem;
        color: #a7f3d0;
        margin-right: 6px;
        margin-top: 4px;
        margin-bottom: 6px;
    }

    /* Agent Breakdown Styling */
    .agent-stage-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .stage-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #10b981;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .pill-kw {
        display: inline-block;
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #6ee7b7;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px 5px 4px 0;
    }
    .chip-audio {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #93c5fd;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        margin: 2px 5px 4px 0;
    }
    .chip-meta {
        display: inline-block;
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.35);
        color: #d8b4fe;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        margin: 2px 5px 4px 0;
    }
    .candidate-card {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .vetting-card-approved {
        background: rgba(16, 185, 129, 0.07);
        border-left: 4px solid #10b981;
        border-top: 1px solid rgba(16, 185, 129, 0.25);
        border-right: 1px solid rgba(16, 185, 129, 0.25);
        border-bottom: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 10px;
    }
    .vetting-card-rejected {
        background: rgba(239, 68, 68, 0.07);
        border-left: 4px solid #ef4444;
        border-top: 1px solid rgba(239, 68, 68, 0.25);
        border-right: 1px solid rgba(239, 68, 68, 0.25);
        border-bottom: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 10px;
    }
    .badge-approved {
        background: #10b981;
        color: #064e3b;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 10px;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-rejected {
        background: #ef4444;
        color: #450a0a;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 10px;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .subtext {
        font-size: 0.83rem;
        color: #9ca3af;
        margin-top: 3px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with status and curated vibe categories
with st.sidebar:
    st.header("🎵 DrakeAI System")
    
    # Check backend health
    try:
        health_res = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if health_res.status_code == 200:
            health_data = health_res.json()
            model_name = health_data.get('gemini_model') or health_data.get('ollama_model') or 'gemini'
            provider = health_data.get('llm_provider', 'Gemini').capitalize()
            st.caption(f"LLM: `{provider} ({model_name})` | Embeddings: `{health_data.get('embedding_dim')}d`")
        else:
            st.warning("Backend Degraded")
    except Exception:
        st.error("● Backend Offline")
        st.caption("Start backend with: `uvicorn backend.main:app --reload`")

    st.divider()
    st.subheader("🦉 Drake Persona & Intelligence")
    st.markdown("""
    - 🎙️ **Authentic Voice**: Drake reflects on his own tracks, lyrics, and mindstate.
    - 🔍 **Dynamic Semantic Router**: Intent expansion with deep emotional nuances.
    - 🎛️ **Acoustic Profiling**: Valence, energy, BPM tempo, and danceability targets.
    - ⚖️ **Parent-Child Vetting**: Context judge evaluates full parent song lyrics.
    """)

    st.divider()
    st.subheader("💡 Example Prompts")
    example_prompts = [
        "What are the top 3 saddest Drake songs?",
        "Show me high-energy club bangers.",
        "Show me introspective Drake lyrics from More Life.",
        "Find late-night songs with slow tempo.",
        "What are Drake's most guarded, paranoid lyrics?",
    ]
    for p in example_prompts:
        if st.button(p, key=f"btn_{p[:15]}"):
            st.session_state.selected_prompt = p

# App Header
st.title("🎵 Drake Lyric & Audio Intelligence")
st.caption("A hybrid semantic RAG agent navigating Drake's discography, stanzas, and albums.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def render_stage1_query_analysis(qa: dict):
    """Renders Stage 1: Key Word Extraction, Audio Feature Creation, & Metadata Limits."""
    if not qa:
        return

    st.markdown('<div class="stage-title">🎯 Stage 1: Query Analysis & Acoustic Profiling</div>', unsafe_allow_html=True)

    # 1. Key Word Extraction
    keywords = qa.get("keywords", [])
    if keywords:
        st.markdown("**🔑 Extracted Keywords & Concepts:**")
        pills = "".join(f'<span class="pill-kw">{kw}</span>' for kw in keywords)
        st.markdown(pills, unsafe_allow_html=True)

    # 2. Audio Feature Creation
    af = qa.get("audio_feature_creation", {})
    targets = af.get("targets", {})
    filters = af.get("filters", {})
    sort_by = af.get("sort_by")

    audio_chips = []
    if targets:
        for k, v in targets.items():
            label = k.replace("target_", "").capitalize()
            if "valence" in k:
                desc = "Melancholic" if v < 0.4 else ("Euphoric" if v > 0.65 else "Neutral")
                audio_chips.append(f'<span class="chip-audio">🎯 Target {label}: {v} ({desc})</span>')
            elif "energy" in k:
                desc = "Subdued" if v < 0.45 else ("High-Energy" if v > 0.7 else "Mid")
                audio_chips.append(f'<span class="chip-audio">⚡ Target {label}: {v} ({desc})</span>')
            elif "tempo" in k:
                audio_chips.append(f'<span class="chip-audio">⏱️ Target Tempo: {v} BPM</span>')
            else:
                audio_chips.append(f'<span class="chip-audio">🎯 Target {label}: {v}</span>')

    if filters:
        for k, v in filters.items():
            audio_chips.append(f'<span class="chip-audio">🔒 Filter {k}: {v}</span>')

    if sort_by:
        audio_chips.append(f'<span class="chip-audio">📊 Sort: {sort_by}</span>')

    if audio_chips:
        st.markdown("**🎧 Audio Feature Creation (Acoustic Targets & Constraints):**")
        st.markdown("".join(audio_chips), unsafe_allow_html=True)

    # 3. Metadata Limits
    meta = qa.get("metadata_limits", {})
    meta_chips = []
    limit = meta.get("limit")
    if limit is not None:
        meta_chips.append(f'<span class="chip-meta">🔢 Limit: {limit} tracks</span>')
    album = meta.get("album_name")
    if album:
        meta_chips.append(f'<span class="chip-meta">💿 Album: {album}</span>')
    y_before = meta.get("release_year_before")
    if y_before:
        meta_chips.append(f'<span class="chip-meta">📅 Released Before: {y_before}</span>')
    y_after = meta.get("release_year_after")
    if y_after:
        meta_chips.append(f'<span class="chip-meta">📅 Released After: {y_after}</span>')

    if meta_chips:
        st.markdown("**📋 Metadata Limits & Scoping:**")
        st.markdown("".join(meta_chips), unsafe_allow_html=True)

    # Expanded Semantic Statement preview
    sq = qa.get("semantic_query")
    if sq and sq != qa.get("prompt"):
        with st.expander("🔍 Expanded Semantic Query Statement", expanded=False):
            st.caption(sq)


def render_stage_sql_generation(sql_data: dict):
    """Renders Stage 1.5: Dynamic Text-to-SQL Agent Query & Strategy."""
    if not sql_data or not sql_data.get("query"):
        return

    st.markdown('<div class="stage-title">🛠️ Stage 2: Dynamic SQL Generation (Text-to-SQL Agent)</div>', unsafe_allow_html=True)
    strategy = sql_data.get("strategy") or "Dynamic query generated based on user intent and metadata."
    q_type = sql_data.get("query_type") or "hybrid_semantic"
    query = sql_data.get("query", "")
    params = sql_data.get("params", {})

    st.markdown(f'<div class="subtext"><b>Intent Strategy:</b> {strategy} &nbsp;|&nbsp; <b>Query Type:</b> <code>{q_type}</code></div>', unsafe_allow_html=True)
    st.code(query.strip(), language="sql")
    if params:
        st.caption(f"**Bound Parameters:** `{params}`")


def render_stage2_pulled_songs(songs: list):
    """Renders Stage 2: Initial Candidates Pulled from Context Graph."""
    if not songs:
        return

    st.markdown('<div class="stage-title">📥 Stage 3: Pulled Candidate Songs (Database Retrieval)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtext">Retrieved <b>{len(songs)}</b> candidate tracks via dynamic SQL context graph retrieval before parent track vetting:</div>', unsafe_allow_html=True)

    for idx, s in enumerate(songs, start=1):
        tname = s.get("track_name", "Unknown")
        aname = s.get("album_name", "Unknown")
        year = f" ({s['release_date']})" if s.get("release_date") else ""
        sim = s.get("similarity")
        hybrid = s.get("hybrid_score")
        snippet = s.get("lyric_snippet", "")

        stat_pills = []
        if sim is not None:
            stat_pills.append(f'<span class="audio-badge">Cosine Sim: {int(sim * 100)}%</span>')
        if hybrid is not None:
            stat_pills.append(f'<span class="audio-badge">🎯 Hybrid Score: {int(hybrid * 100)}%</span>')
        if s.get("valence") is not None:
            stat_pills.append(f'<span class="audio-badge">📉 Val: {s["valence"]}</span>')
        if s.get("energy") is not None:
            stat_pills.append(f'<span class="audio-badge">⚡ Energy: {s["energy"]}</span>')
        if s.get("tempo") is not None:
            stat_pills.append(f'<span class="audio-badge">⏱️ {s["tempo"]} BPM</span>')

        st.markdown(f"""
        <div class="candidate-card">
            <b>#{idx} {tname}</b> — <i>{aname}</i>{year}<br/>
            {"".join(stat_pills)}
            <div style="font-size: 0.82rem; font-style: italic; color: #d1d5db; margin-top: 4px;">
                "{snippet}..."
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_stage3_vetting_agent(vetting: dict):
    """Renders Stage 3: Parent-Child Vetting Agent Evaluations & Decisions."""
    if not vetting:
        return

    st.markdown('<div class="stage-title">⚖️ Stage 4: Parent-Child Vetting Agent (Evaluation Judge)</div>', unsafe_allow_html=True)
    decisions = vetting.get("decisions", [])
    retries = vetting.get("retry_count", 0)

    retry_info = f" • (Retries triggered: {retries})" if retries > 0 else ""
    st.markdown(f'<div class="subtext">The Vetting Judge pulls <b>full parent song lyrics</b> into context to verify each candidate against the query tone and acoustics{retry_info}:</div>', unsafe_allow_html=True)

    if not decisions:
        rat = vetting.get("primary_rationale") or "Context vetted."
        st.info(rat)
        return

    for d in decisions:
        status = d.get("status", "APPROVED")
        is_match = d.get("is_match", True)
        card_class = "vetting-card-approved" if is_match else "vetting-card-rejected"
        badge_class = "badge-approved" if is_match else "badge-rejected"
        badge_text = "APPROVED" if is_match else "REJECTED"
        icon = "✅" if is_match else "❌"

        tname = d.get("track_name", "Unknown Track")
        aname = d.get("album_name", "")
        conf = int(d.get("confidence", 1.0) * 100)
        reason = d.get("reason", "No justification provided.")

        af_bits = []
        if d.get("valence") is not None:
            af_bits.append(f"Valence: {d['valence']}")
        if d.get("energy") is not None:
            af_bits.append(f"Energy: {d['energy']}")
        if d.get("tempo") is not None:
            af_bits.append(f"Tempo: {d['tempo']} BPM")
        af_str = f" • {' | '.join(af_bits)}" if af_bits else ""

        st.markdown(f"""
        <div class="{card_class}">
            <div>
                <span class="{badge_class}">{icon} {badge_text}</span>
                <b>{tname}</b> ({aname}) • <b>{conf}% Confidence</b>{af_str}
            </div>
            <div style="font-size: 0.86rem; color: #e5e7eb; margin-top: 6px;">
                <b>Vetting Rationale:</b> {reason}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_agent_breakdown(trace: dict):
    """Renders complete agent reasoning breakdown across all stages."""
    if not trace:
        st.caption("No agent trace available for this response.")
        return

    st.markdown('<div class="agent-stage-box">', unsafe_allow_html=True)

    # 1. Query Analysis & Acoustic Profiling
    render_stage1_query_analysis(trace.get("query_analysis", {}))
    st.divider()

    # 2. Dynamic Text-to-SQL Generation
    if trace.get("sql_generation"):
        render_stage_sql_generation(trace.get("sql_generation", {}))
        st.divider()

    # 3. Pulled Candidate Songs
    render_stage2_pulled_songs(trace.get("pulled_songs", []))
    st.divider()

    # 4. Parent-Child Vetting Agent
    render_stage3_vetting_agent(trace.get("vetting_agent", {}))

    # 5. Musicological Synthesis Preview
    thematic = trace.get("thematic_analysis")
    if thematic:
        st.divider()
        st.markdown('<div class="stage-title">🦉 Stage 5: Drake\'s Thematic Reflection</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.88rem; color: #d1d5db; font-style: italic;">{thematic}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)



# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show collapsible agent breakdown for prior assistant responses
        if message.get("agent_trace"):
            with st.expander("🧠 Agent Thinking Breakdown (Query Analysis → Pulled Songs → Vetting)", expanded=False):
                render_agent_breakdown(message["agent_trace"])

        st.markdown(message["content"], unsafe_allow_html=True)

# Check if an example prompt was clicked
prefill_prompt = st.session_state.pop("selected_prompt", None)

# User input
prompt = st.chat_input("Ask about Drake lyrics, mood, themes, or albums...")
if prefill_prompt and not prompt:
    prompt = prefill_prompt

if prompt:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend with streaming & live thinking visualization
    with st.chat_message("assistant"):
        answer = ""
        sources = []
        agent_trace = None

        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # Live thinking container
        with st.status("🧠 DrakeAI Agent Pipeline Thinking...", expanded=True) as status_box:
            stage1_placeholder = st.empty()
            stagesql_placeholder = st.empty()
            stage2_placeholder = st.empty()
            stage3_placeholder = st.empty()
            stage4_placeholder = st.empty()

            try:
                # Stream agent execution events from backend
                stream_res = requests.post(
                    f"{BACKEND_URL}/api/chat/stream",
                    json={"prompt": prompt, "history": history_payload},
                    stream=True,
                    timeout=90
                )

                if stream_res.status_code == 200:
                    for line in stream_res.iter_lines(decode_unicode=True):
                        if not line or not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                            step = event.get("step")

                            if step == "query_analysis":
                                status_box.write("🔍 **Stage 1 Complete:** Query keywords extracted & acoustic profile generated")
                                with stage1_placeholder.container():
                                    render_stage1_query_analysis(event)

                            elif step == "sql_generation":
                                status_box.write("🛠️ **Stage 2 Complete:** Dynamic Text-to-SQL query synthesized")
                                with stagesql_placeholder.container():
                                    render_stage_sql_generation(event)

                            elif step == "pulled_songs":
                                status_box.write("📥 **Stage 3 Complete:** Pulled candidate tracks from PostgreSQL context graph")
                                with stage2_placeholder.container():
                                    render_stage2_pulled_songs(event.get("pulled_songs", []))

                            elif step == "vetting_agent":
                                status_box.write("⚖️ **Stage 4 Complete:** Parent-child vetting evaluations complete")
                                with stage3_placeholder.container():
                                    render_stage3_vetting_agent(event)

                            elif step == "reasoning":
                                status_box.write("🦉 **Stage 5 Complete:** Drake's thematic reflection synthesized")
                                thematic = event.get("thematic_analysis")
                                if thematic:
                                    with stage4_placeholder.container():
                                        st.markdown('<div class="stage-title">🦉 Stage 5: Drake\'s Thematic Reflection</div>', unsafe_allow_html=True)
                                        st.markdown(f'<div style="font-size: 0.88rem; color: #d1d5db; font-style: italic;">{thematic}</div>', unsafe_allow_html=True)

                            elif step == "complete":
                                answer = event.get("response", "No response generated.")
                                sources = event.get("sources", [])
                                agent_trace = event.get("agent_trace")

                            elif step == "error":
                                answer = f"Pipeline error: {event.get('error')}"

                        except Exception as parse_err:
                            pass

                    # Collapse status upon completion
                    status_box.update(
                        label="🧠 Agent Thinking Breakdown (Query Analysis → SQL Generation → Pulled Songs → Vetting)",
                        state="complete",
                        expanded=False
                    )
                else:
                    # Fallback to standard chat endpoint if stream not 200
                    fallback_res = requests.post(
                        f"{BACKEND_URL}/api/chat",
                        json={"prompt": prompt, "history": history_payload},
                        timeout=90
                    )
                    if fallback_res.status_code == 200:
                        data = fallback_res.json()
                        answer = data.get("response", "")
                        sources = data.get("sources", [])
                        agent_trace = data.get("agent_trace")
                        if agent_trace:
                            with stage1_placeholder.container():
                                render_agent_breakdown(agent_trace)
                    else:
                        answer = f"Error: Backend returned HTTP {fallback_res.status_code}: {fallback_res.text}"

                    status_box.update(
                        label="🧠 Agent Thinking Breakdown (Query Analysis → Pulled Songs → Vetting)",
                        state="complete",
                        expanded=False
                    )

            except Exception as e:
                status_box.update(
                    label="⚠️ Error Connecting to Backend Pipeline",
                    state="error",
                    expanded=False
                )
                answer = f"Error connecting to backend: {e}"
                sources = []

        # Unique anchor placed immediately at the start of the generated response
        response_anchor_id = f"response-start-{len(st.session_state.messages)}"
        st.markdown(f'<div id="{response_anchor_id}" class="response-anchor"></div>', unsafe_allow_html=True)

        # Render synthesized answer with integrated artwork & Spotify links
        if answer:
            st.markdown(answer, unsafe_allow_html=True)

        # Smoothly bring the view to the start of the response generated, not at the bottom
        components.html(
            f"""
            <script>
                (function() {{
                    const targetId = "{response_anchor_id}";
                    function scrollToResponseStart() {{
                        try {{
                            const doc = window.parent.document;
                            const el = doc.getElementById(targetId);
                            if (el) {{
                                el.scrollIntoView({{ behavior: "smooth", block: "start" }});
                            }}
                        }} catch (e) {{
                            console.error("Auto-scroll error:", e);
                        }}
                    }}
                    // Staggered calls to ensure DOM updates and status container collapse have finished
                    setTimeout(scrollToResponseStart, 50);
                    setTimeout(scrollToResponseStart, 250);
                    setTimeout(scrollToResponseStart, 500);
                }})();
            </script>
            """,
            height=0,
            width=0
        )

    # Save assistant message with sources & agent_trace to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "agent_trace": agent_trace
    })