import json
import requests
import streamlit as st

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
    .spotify-btn {
        display: inline-block;
        background-color: #1DB954;
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        text-decoration: none;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
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
    st.subheader("🎧 Vibe & Lyric Categories")
    st.markdown("""
    - 🌙 **Late-Night Confessional**
    - 🏆 **Triumphant Flex**
    - 🛡️ **Paranoid & Guarded**
    - ⏳ **Time-Stamp Introspection**
    - 💔 **Toxic & Petty**
    - 🌴 **Global Groove / Island Infusion**
    - 📻 **Pop Crossover / Radio R&B**
    - 💥 **Hard-Hitting / Mob Tie**
    - 🤝 **Crew Loyalty & Brotherhood**
    - 🪩 **The Club Anthem**
    """)

    st.divider()
    st.subheader("💡 Example Prompts")
    example_prompts = [
        "Show me introspective Drake lyrics from More Life.",
        "Find late-night confessional songs released before 2018.",
        "What are Drake's hardest-hitting mob tie lyrics?",
        "Show me lyrics about heartbreak and loyalty.",
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

# Helper to render sources
def render_sources(sources):
    if not sources:
        return
    st.markdown("#### 📚 Sources & Attribution")
    for src in sources:
        track_name = src.get("track_name", "Unknown Track")
        album_name = src.get("album_name", "Unknown Album")
        rel_date = src.get("release_date", "")
        year_str = f" ({rel_date[:4]})" if rel_date else ""
        art_url = src.get("album_art_url")
        spotify_url = src.get("spotify_url")
        stanzas = src.get("quoted_stanzas", [])
        feel = src.get("personal_feel")
        rationale = src.get("match_rationale")

        cols = st.columns([1, 4])
        with cols[0]:
            if art_url:
                st.image(art_url, use_container_width=True)
            else:
                st.markdown("💿")
        with cols[1]:
            feel_badge = f" • *{feel}*" if feel else ""
            st.markdown(f"**{track_name}** — *{album_name}*{year_str}{feel_badge}")
            
            if rationale:
                st.markdown(f"""
                <div class="rationale-box">
                    <strong>💡 Why this matches:</strong><br/>
                    {rationale}
                </div>
                """, unsafe_allow_html=True)

            if spotify_url:
                st.markdown(f"[▶ Listen on Spotify]({spotify_url})")
            for chunk in stanzas[:2]:
                st.markdown(f"> *\"{chunk.strip()}\"*")
        st.divider()


# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Quoted Tracks & Artwork", expanded=False):
                render_sources(message["sources"])

# Check if an example prompt was clicked
prefill_prompt = st.session_state.pop("selected_prompt", None)

# User input
prompt = st.chat_input("Ask about Drake lyrics, vibes, or albums...")
if prefill_prompt and not prompt:
    prompt = prefill_prompt

if prompt:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching stanzas and synthesizing answer..."):
            try:
                # History excluding current user message
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                response = requests.post(
                    f"{BACKEND_URL}/api/chat",
                    json={"prompt": prompt, "history": history_payload},
                    timeout=90
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("response", "No response received.")
                    sources = data.get("sources", [])
                else:
                    answer = f"Error: Backend returned HTTP {response.status_code}: {response.text}"
                    sources = []
            except Exception as e:
                answer = f"Error connecting to backend: {e}"
                sources = []

            st.markdown(answer)
            if sources:
                with st.expander("View Quoted Tracks & Artwork", expanded=True):
                    render_sources(sources)

    # Save assistant message with sources to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })