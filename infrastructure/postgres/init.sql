-- PostgreSQL initialization for RAG Chatbot
-- Chat session management tables

-- New schema: numeric id as PK, unique session name from client
CREATE TABLE IF NOT EXISTS chat_sessions (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    state VARCHAR(50) NOT NULL DEFAULT 'initial',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    pending_questions JSONB DEFAULT '[]',
    clarifications JSONB DEFAULT '{}',
    current_thread_id BIGINT NOT NULL DEFAULT 0,
    max_clarification_rounds INT NOT NULL DEFAULT 2
);

-- Messages table: stores all messages in chat sessions
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    stage VARCHAR(50) NOT NULL DEFAULT 'user_question',
    thread_id BIGINT NOT NULL DEFAULT 0,
    round INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_state ON chat_sessions(state);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(session_id, thread_id, created_at);

-- Thread summaries: one row per (session_id, thread_id)
CREATE TABLE IF NOT EXISTS chat_thread_summaries (
    id BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    thread_id BIGINT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_thread_summaries_session ON chat_thread_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_thread_summaries_created_at ON chat_thread_summaries(created_at);

-- Trigger to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_chat_sessions_updated_at BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Audit schema (if needed for future use)
CREATE SCHEMA IF NOT EXISTS audit;
