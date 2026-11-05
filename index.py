"""Vector index in a single SQLite file (sqlite-vec), synced incrementally one file at a time.

Only files whose sha256 changed are re-chunked and re-embedded, and files deleted from the
docs folder are removed from the index. Changing the embedding model, the docs folder, or the
chunking settings rebuilds the whole index, so vectors from different setups never get mixed.
"""
import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np
import sqlite_vec

from chunker import MAX_CHARS, chunk_markdown

SCHEMA_VERSION = "1"  # bump when the tables or chunking logic change, to force a rebuild
EMBED_BATCH = 64      # chunks per /api/embed request


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    return db


def has_vectors(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name = 'vec_chunks'").fetchone() is not None


def reset(db):
    db.executescript("""
        DROP TABLE IF EXISTS vec_chunks;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS files;
        DELETE FROM meta;
        CREATE TABLE files (
            path TEXT PRIMARY KEY,   -- relative to the docs folder
            sha256 TEXT NOT NULL,
            frontmatter TEXT         -- raw YAML frontmatter, kept as metadata
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,  -- same value as vec_chunks.rowid
            path TEXT NOT NULL REFERENCES files(path),
            heading TEXT NOT NULL,   -- e.g. "Decision > Sizing"
            text TEXT NOT NULL       -- the embedded text, including the path/heading prefix
        );
        CREATE INDEX chunks_path ON chunks(path);
    """)


def doc_files(docs_dir):
    """Every .md file under docs_dir, skipping hidden folders such as .git, .claude, .github."""
    return sorted(
        p for p in docs_dir.rglob("*.md")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(docs_dir).parts)
    )


def delete_file(db, rel):
    if has_vectors(db):
        db.execute("DELETE FROM vec_chunks WHERE rowid IN (SELECT id FROM chunks WHERE path = ?)", (rel,))
    db.execute("DELETE FROM chunks WHERE path = ?", (rel,))
    db.execute("DELETE FROM files WHERE path = ?", (rel,))


def sync(db, docs_dir, embed, embed_model, force=False):
    """Bring the index up to date with docs_dir. `embed(texts)` returns one vector per text."""
    docs_dir = Path(docs_dir).expanduser().resolve()
    if not docs_dir.is_dir():
        raise SystemExit(f"Docs folder not found: {docs_dir}")

    wanted = {"schema": SCHEMA_VERSION, "embed_model": embed_model,
              "docs_dir": str(docs_dir), "max_chars": str(MAX_CHARS)}
    if force or dict(db.execute("SELECT key, value FROM meta")) != wanted:
        reset(db)
        db.executemany("INSERT INTO meta VALUES (?, ?)", wanted.items())
        db.commit()

    stored = dict(db.execute("SELECT path, sha256 FROM files"))
    current, todo = set(), []
    for f in doc_files(docs_dir):
        rel = f.relative_to(docs_dir).as_posix()
        data = f.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        current.add(rel)
        if stored.get(rel) != sha:
            todo.append((rel, sha, data.decode("utf-8", errors="replace")))

    removed = sorted(stored.keys() - current)
    for rel in removed:
        delete_file(db, rel)
    db.commit()

    start, embedded = time.perf_counter(), 0
    for n, (rel, sha, text) in enumerate(todo, 1):
        frontmatter, chunks = chunk_markdown(rel, text)
        texts = [t for _, t in chunks]
        vectors = [v for i in range(0, len(texts), EMBED_BATCH) for v in embed(texts[i:i + EMBED_BATCH])]
        if vectors and not has_vectors(db):
            db.execute(f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
                       f"embedding float[{len(vectors[0])}] distance_metric=cosine)")
        # Replace this file's rows in one transaction, so an interrupted sync leaves no half-indexed file.
        delete_file(db, rel)
        db.execute("INSERT INTO files VALUES (?, ?, ?)", (rel, sha, frontmatter))
        for (heading, chunk_text), vec in zip(chunks, vectors):
            cur = db.execute("INSERT INTO chunks (path, heading, text) VALUES (?, ?, ?)", (rel, heading, chunk_text))
            db.execute("INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                       (cur.lastrowid, np.asarray(vec, dtype=np.float32).tobytes()))
        db.commit()
        embedded += len(chunks)
        print(f"  [{n}/{len(todo)}] {rel} ({len(chunks)} chunks)", flush=True)

    total_files, = db.execute("SELECT count(*) FROM files").fetchone()
    total_chunks, = db.execute("SELECT count(*) FROM chunks").fetchone()
    print(f"Index: {total_files} files, {total_chunks} chunks "
          f"({len(todo)} files re-embedded, {len(removed)} removed)")
    if todo:
        print(f"Embedded {embedded} chunks in {time.perf_counter() - start:.1f}s")


def search(db, query_vector, k=3):
    """Return the k nearest chunks as (path, heading, text, cosine_distance) tuples."""
    if not has_vectors(db):
        return []
    return db.execute("""
        WITH knn AS (
            SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?
        )
        SELECT c.path, c.heading, c.text, knn.distance
        FROM knn JOIN chunks c ON c.id = knn.rowid
        ORDER BY knn.distance
    """, (np.asarray(query_vector, dtype=np.float32).tobytes(), k)).fetchall()
