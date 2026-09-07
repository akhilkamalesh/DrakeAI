import json
import random
import os

# Get directory of this script to write files relatively
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Lazy model loader
model = None

# Helper to generate embedding (real or mock 1024-dimensional vector)
def generate_embedding(seed_val, text=None):
    global model
    if HAS_SENTENCE_TRANSFORMERS and text is not None:
        if model is None:
            print("Loading sentence-transformers 'BAAI/bge-large-en-v1.5' model...")
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        # Generate real embedding (returns a numpy array)
        emb = model.encode(text)
        # Convert to standard Python float list
        return [float(x) for x in emb]
    else:
        # Fallback to random 1024-dimensional vector (matches BAAI/bge-large-en-v1.5)
        random.seed(seed_val)
        vec = [random.uniform(-0.1, 0.1) for _ in range(1024)]
        norm = sum(x*x for x in vec) ** 0.5
        return [x / norm for x in vec]

albums = [
    {
        "id": "4eLPsYPBmKkvIiYbbCY2Nq",
        "name": "Take Care",
        "album_type": "album",
        "images_url": "https://open.spotify.com/image/take_care_cover",
        "release_date": "2011-11-15",
        "total_tracks": 18
    },
    {
        "id": "7tyx094Wmq3s71575UaU6t",
        "name": "Nothing Was the Same",
        "album_type": "album",
        "images_url": "https://open.spotify.com/image/nwts_cover",
        "release_date": "2013-09-24",
        "total_tracks": 13
    },
    {
        "id": "40GMAh5YJ6w58o6V2o77Qo",
        "name": "Views",
        "album_type": "album",
        "images_url": "https://open.spotify.com/image/views_cover",
        "release_date": "2016-04-29",
        "total_tracks": 20
    },
    {
        "id": "1t5bM41M4k2sXWw8UeJ4Ld",
        "name": "Scorpion",
        "album_type": "album",
        "images_url": "https://open.spotify.com/image/scorpion_cover",
        "release_date": "2018-06-29",
        "total_tracks": 25
    }
]

tracks = [
    {
        "id": "27tNWlJeNuIJbY0OM63Z2W",
        "album_id": "4eLPsYPBmKkvIiYbbCY2Nq",
        "disc_number": 1,
        "name": "Headlines",
        "lyrics": "I might be too strung out on compliments, overdosed on confidence\nStarted not to give a fuck and stopped fearing the consequence\nDrinking every night because we drink to my accomplishments\nFaded way too long, I'm floating in and out of consciousness",
        "external_urls": {"spotify": "https://open.spotify.com/track/27tNWlJeNuIJbY0OM63Z2W"}
    },
    {
        "id": "3jq620G6eB96ZJ5G1Xn85s",
        "album_id": "7tyx094Wmq3s71575UaU6t",
        "disc_number": 1,
        "name": "Started From the Bottom",
        "lyrics": "Started from the bottom, now we're here\nStarted from the bottom, now my whole team fucking here\nStarted from the bottom, now we're here\nStarted from the bottom, now my whole team here, yuh",
        "external_urls": {"spotify": "https://open.spotify.com/track/3jq620G6eB96ZJ5G1Xn85s"}
    },
    {
        "id": "0G21yYKMzoeg4glqEqjNvc",
        "album_id": "40GMAh5YJ6w58o6V2o77Qo",
        "disc_number": 1,
        "name": "Hotline Bling",
        "lyrics": "You used to call me on my cell phone\nLate night when you need my love\nCall me on my cell phone\nLate night when you need my love\nAnd I know when that hotline bling\nThat can only mean one thing",
        "external_urls": {"spotify": "https://open.spotify.com/track/0G21yYKMzoeg4glqEqjNvc"}
    },
    {
        "id": "6DCZqySq6xGg8H3aE5V5u5",
        "album_id": "1t5bM41M4k2sXWw8UeJ4Ld",
        "disc_number": 1,
        "name": "God's Plan",
        "lyrics": "And they wishin' and wishin' and wishin' and wishin'\nThey wishin' bad things on me\nBad things on me\nI hold back, sometimes I won't, yuh\nI feel good, sometimes I don't, ayy, don't",
        "external_urls": {"spotify": "https://open.spotify.com/track/6DCZqySq6xGg8H3aE5V5u5"}
    }
]

audio_features = [
    {
        "song_id": "27tNWlJeNuIJbY0OM63Z2W",
        "acousticness": 0.35,
        "danceability": 0.63,
        "energy": 0.57,
        "instrumentalness": 0.0,
        "key": 6,
        "liveness": 0.08,
        "loudness": -7.1,
        "mode": 0,
        "speechiness": 0.11,
        "tempo": 152.0,
        "time_signature": 4,
        "valence": 0.44
    },
    {
        "song_id": "3jq620G6eB96ZJ5G1Xn85s",
        "acousticness": 0.03,
        "danceability": 0.79,
        "energy": 0.52,
        "instrumentalness": 0.0,
        "key": 8,
        "liveness": 0.16,
        "loudness": -7.8,
        "mode": 1,
        "speechiness": 0.16,
        "tempo": 86.0,
        "time_signature": 4,
        "valence": 0.51
    },
    {
        "song_id": "0G21yYKMzoeg4glqEqjNvc",
        "acousticness": 0.01,
        "danceability": 0.89,
        "energy": 0.62,
        "instrumentalness": 0.0001,
        "key": 2,
        "liveness": 0.05,
        "loudness": -4.9,
        "mode": 1,
        "speechiness": 0.06,
        "tempo": 135.0,
        "time_signature": 4,
        "valence": 0.55
    },
    {
        "song_id": "6DCZqySq6xGg8H3aE5V5u5",
        "acousticness": 0.03,
        "danceability": 0.75,
        "energy": 0.45,
        "instrumentalness": 0.0001,
        "key": 7,
        "liveness": 0.55,
        "loudness": -9.2,
        "mode": 1,
        "speechiness": 0.10,
        "tempo": 77.1,
        "time_signature": 4,
        "valence": 0.36
    }
]

stanzas = [
    # Headlines stanzas
    {
        "song_id": "27tNWlJeNuIJbY0OM63Z2W",
        "lyric_chunk": "I might be too strung out on compliments, overdosed on confidence\nStarted not to give a fuck and stopped fearing the consequence",
        "chunk_index": 0,
        "personal_feel": "Introspective"
    },
    {
        "song_id": "27tNWlJeNuIJbY0OM63Z2W",
        "lyric_chunk": "Drinking every night because we drink to my accomplishments\nFaded way too long, I'm floating in and out of consciousness",
        "chunk_index": 1,
        "personal_feel": "Late Night Drive"
    },
    # Started From the Bottom
    {
        "song_id": "3jq620G6eB96ZJ5G1Xn85s",
        "lyric_chunk": "Started from the bottom, now we're here\nStarted from the bottom, now my whole team fucking here",
        "chunk_index": 0,
        "personal_feel": "Hype"
    },
    # Hotline Bling
    {
        "song_id": "0G21yYKMzoeg4glqEqjNvc",
        "lyric_chunk": "You used to call me on my cell phone\nLate night when you need my love",
        "chunk_index": 0,
        "personal_feel": "Late Night Drive"
    },
    # God's Plan
    {
        "song_id": "6DCZqySq6xGg8H3aE5V5u5",
        "lyric_chunk": "I hold back, sometimes I won't, yuh\nI feel good, sometimes I don't, ayy, don't",
        "chunk_index": 0,
        "personal_feel": "Introspective"
    }
]

# Generate stanzas embeddings dynamically
for i, s in enumerate(stanzas):
    print(f"Generating embedding for stanza {i} ({s['song_id']})...")
    s["embedding"] = generate_embedding(100 + i, s["lyric_chunk"])

# Write JSON Files
with open(os.path.join(script_dir, "albums.json"), "w") as f:
    json.dump(albums, f, indent=4)

with open(os.path.join(script_dir, "tracks.json"), "w") as f:
    json.dump(tracks, f, indent=4)

with open(os.path.join(script_dir, "audio_features.json"), "w") as f:
    json.dump(audio_features, f, indent=4)

with open(os.path.join(script_dir, "stanzas.json"), "w") as f:
    json.dump(stanzas, f, indent=4)

# Generate SQL seed file
sql_lines = [
    "-- Mock data seeding for Drake RAG database",
    "TRUNCATE album, track, audio_feature, stanza CASCADE;\n"
]

# Insert albums
for a in albums:
    album_name_escaped = a['name'].replace("'", "''")
    sql_lines.append(
        f"INSERT INTO album (id, name, album_type, images_url, release_date, total_tracks) "
        f"VALUES ('{a['id']}', '{album_name_escaped}', '{a['album_type']}', '{a['images_url']}', '{a['release_date']}', {a['total_tracks']}) "
        f"ON CONFLICT (id) DO NOTHING;"
    )

sql_lines.append("")

# Insert tracks
for t in tracks:
    track_name_escaped = t['name'].replace("'", "''")
    lyrics_escaped = t['lyrics'].replace("'", "''")
    ext_url_json = json.dumps(t['external_urls'])
    sql_lines.append(
        f"INSERT INTO track (id, album_id, disc_number, name, lyrics, external_urls) "
        f"VALUES ('{t['id']}', '{t['album_id']}', {t['disc_number']}, '{track_name_escaped}', '{lyrics_escaped}', '{ext_url_json}'::jsonb) "
        f"ON CONFLICT (id) DO NOTHING;"
    )

sql_lines.append("")

# Insert audio features
for af in audio_features:
    sql_lines.append(
        f"INSERT INTO audio_feature (song_id, acousticness, danceability, energy, instrumentalness, \"key\", liveness, loudness, \"mode\", speechiness, tempo, time_signature, valence) "
        f"VALUES ('{af['song_id']}', {af['acousticness']}, {af['danceability']}, {af['energy']}, {af['instrumentalness']}, {af['key']}, {af['liveness']}, {af['loudness']}, {af['mode']}, {af['speechiness']}, {af['tempo']}, {af['time_signature']}, {af['valence']}) "
        f"ON CONFLICT (song_id) DO NOTHING;"
    )

sql_lines.append("")

# Insert stanzas
for i, s in enumerate(stanzas):
    chunk_escaped = s['lyric_chunk'].replace("'", "''")
    embedding_str = "[" + ",".join(str(val) for val in s['embedding']) + "]"
    personal_feel_escaped = s['personal_feel'].replace("'", "''") if s['personal_feel'] else None
    
    personal_feel_val = f"'{personal_feel_escaped}'" if personal_feel_escaped else "NULL"
    sql_lines.append(
        f"INSERT INTO stanza (id, song_id, lyric_chunk, chunk_index, embedding, personal_feel) "
        f"VALUES ({i+1}, '{s['song_id']}', '{chunk_escaped}', {s['chunk_index']}, '{embedding_str}'::vector, {personal_feel_val}) "
        f"ON CONFLICT (id) DO NOTHING;"
    )

with open(os.path.join(script_dir, "insert_mock_data.sql"), "w") as f:
    f.write("\n".join(sql_lines))

print("Mock data generated successfully!")
