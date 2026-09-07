-- DDL generated from schema.json
-- Target Database: PostgreSQL with pgvector extension

-- Enable the vector extension (pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create album table
CREATE TABLE IF NOT EXISTS album (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    album_type VARCHAR(50),
    images_url TEXT,
    release_date DATE,
    total_tracks INTEGER
);

COMMENT ON TABLE album IS 'Represents Drake''s albums cataloged from Spotify API';
COMMENT ON COLUMN album.id IS 'Unique Spotify ID for the album';
COMMENT ON COLUMN album.name IS 'Name of the album';
COMMENT ON COLUMN album.album_type IS 'Type of album (e.g., ''album'', ''single'', ''compilation'')';
COMMENT ON COLUMN album.images_url IS 'URL of the album artwork image';
COMMENT ON COLUMN album.release_date IS 'Release date of the album';
COMMENT ON COLUMN album.total_tracks IS 'Total number of tracks on the album';

-- Create track table
CREATE TABLE IF NOT EXISTS track (
    id VARCHAR(255) PRIMARY KEY,
    album_id VARCHAR(255) NOT NULL REFERENCES album(id) ON DELETE CASCADE,
    disc_number INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    lyrics TEXT,
    external_urls JSONB
);

COMMENT ON TABLE track IS 'Represents Drake''s songs cataloged from Spotify API & Musixmatch API';
COMMENT ON COLUMN track.id IS 'Unique Spotify ID for the track';
COMMENT ON COLUMN track.album_id IS 'Foreign key referencing the album table (album.id)';
COMMENT ON COLUMN track.disc_number IS 'Disc number of the track (usually 1 unless a multi-disc release)';
COMMENT ON COLUMN track.name IS 'Name of the track';
COMMENT ON COLUMN track.lyrics IS 'Full lyrics text of the track';
COMMENT ON COLUMN track.external_urls IS 'JSON object containing external URLs (e.g., Spotify links)';

-- Create audio_feature table
CREATE TABLE IF NOT EXISTS audio_feature (
    song_id VARCHAR(255) PRIMARY KEY REFERENCES track(id) ON DELETE CASCADE,
    acousticness REAL,
    danceability REAL,
    energy REAL,
    instrumentalness REAL,
    "key" INTEGER,
    liveness REAL,
    loudness REAL,
    "mode" INTEGER,
    speechiness REAL,
    tempo REAL,
    time_signature INTEGER,
    valence REAL
);

COMMENT ON TABLE audio_feature IS 'Represents audio feature for a drake song';
COMMENT ON COLUMN audio_feature.song_id IS 'Foreign key referencing the track table (track.id)';
COMMENT ON COLUMN audio_feature.acousticness IS 'A confidence measure from 0.0 to 1.0 of whether the track is acoustic';
COMMENT ON COLUMN audio_feature.danceability IS 'Danceability describes how suitable a track is for dancing based on musical elements';
COMMENT ON COLUMN audio_feature.energy IS 'A measure from 0.0 to 1.0 representing a perceptual measure of intensity and activity';
COMMENT ON COLUMN audio_feature.instrumentalness IS 'Predicts whether a track contains no vocals (0.0 to 1.0)';
COMMENT ON COLUMN audio_feature.key IS 'The key the track is in using standard Pitch Class notation (e.g., 0 = C, 1 = C#, 2 = D)';
COMMENT ON COLUMN audio_feature.liveness IS 'Detects the presence of an audience in the recording (0.0 to 1.0)';
COMMENT ON COLUMN audio_feature.loudness IS 'The overall loudness of a track in decibels (dB)';
COMMENT ON COLUMN audio_feature.mode IS 'Modality of a track (1 = major, 0 = minor)';
COMMENT ON COLUMN audio_feature.speechiness IS 'Speechiness detects the presence of spoken words in a track';
COMMENT ON COLUMN audio_feature.tempo IS 'The overall estimated tempo of a track in beats per minute (BPM)';
COMMENT ON COLUMN audio_feature.time_signature IS 'An estimated time signature, specifying how many beats are in each bar';
COMMENT ON COLUMN audio_feature.valence IS 'A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track';

-- Create stanza table
CREATE TABLE IF NOT EXISTS stanza (
    id SERIAL PRIMARY KEY,
    song_id VARCHAR(255) NOT NULL REFERENCES track(id) ON DELETE CASCADE,
    lyric_chunk TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    personal_feel VARCHAR(255)
);

COMMENT ON TABLE stanza IS 'Stores individual lyric chunks optimized for Vector Search and RAG. Will be based of the lyric column in song_id';
COMMENT ON COLUMN stanza.id IS 'Unique auto-incrementing primary key for the stanza';
COMMENT ON COLUMN stanza.song_id IS 'Foreign key referencing the track table (track.id)';
COMMENT ON COLUMN stanza.lyric_chunk IS 'Text content of the lyric stanza/chunk';
COMMENT ON COLUMN stanza.chunk_index IS 'Sequential index of the stanza within the track''s lyrics';
COMMENT ON COLUMN stanza.embedding IS 'Dense vector embedding representing the lyric chunk for semantic search';
COMMENT ON COLUMN stanza.personal_feel IS 'Subjective categorization classification for the stanza (e.g., ''Late Night Drive'', ''Hype'', ''Introspective'')';

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_track_album_id ON track(album_id);
CREATE INDEX IF NOT EXISTS idx_stanza_song_id ON stanza(song_id);

-- Create index for vector similarity search (using HNSW)
CREATE INDEX IF NOT EXISTS idx_stanza_embedding ON stanza USING hnsw (embedding vector_cosine_ops);
