"""Markdown chunking: one chunk per H1-H3 section, never cutting a code block in half.

Each chunk starts with its source path and heading trail, e.g.
"architecture/adr-0018.md > Decision > Sizing", so the embedding knows where the text
came from and the chat model can cite it.
"""
import re

# nomic-embed-text reads at most 2048 tokens. Code and tables can use about one token
# per 2-3 characters, so 2000 characters per chunk stays well inside that limit.
MAX_CHARS = 2000

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
HEADING = re.compile(r"^(#{1,3})[ \t]+(.+?)[ \t#]*$")
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def split_frontmatter(text):
    """Return (frontmatter, body). Frontmatter is the raw YAML text, or "" if there is none."""
    m = FRONTMATTER.match(text)
    return (m.group(1), text[m.end():]) if m else ("", text)


def sections(body):
    """Yield (heading_trail, body_lines) for each H1-H3 section. '#' lines inside code blocks are ignored."""
    levels = [None, None, None]
    lines, fence = [], None
    for line in body.splitlines():
        m = FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)
        elif fence is not None and line.strip().startswith(fence):
            fence = None
        elif fence is None and (h := HEADING.match(line)):
            yield [t for t in levels if t], lines
            depth = len(h.group(1))
            levels[depth - 1:] = [h.group(2)] + [None] * (3 - depth)
            lines = []
            continue
        lines.append(line)
    yield [t for t in levels if t], lines


def blocks(lines):
    """Group lines into code blocks, tables, and paragraphs, which become the units for splitting."""
    out, cur, kind, fence = [], [], None, None

    def flush():
        nonlocal cur, kind
        if cur:
            out.append((kind, cur))
        cur, kind = [], None

    for line in lines:
        if fence is not None:
            cur.append(line)
            if line.strip().startswith(fence):
                fence = None
                flush()
            continue
        if m := FENCE.match(line):
            flush()
            fence, kind, cur = m.group(1), "code", [line]
        elif line.lstrip().startswith("|"):
            if kind != "table":
                flush()
                kind = "table"
            cur.append(line)
        elif not line.strip():
            flush()
        else:
            if kind != "text":
                flush()
                kind = "text"
            cur.append(line)
    flush()  # also closes a code block left unterminated at the end of the section
    return out


def pack(lines, limit, header=(), footer=()):
    """Split lines into groups of at most `limit` chars, repeating header/footer in every group."""
    overhead = sum(len(x) + 1 for x in (*header, *footer))
    room = max(limit - overhead, 200)
    groups, cur, size = [], [], 0
    for line in lines:
        while len(line) > room:  # a single very long line: cut it hard
            groups.append([line[:room]])
            line = line[room:]
        if cur and size + len(line) + 1 > room:
            groups.append(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        groups.append(cur)
    return ["\n".join([*header, *g, *footer]) for g in groups]


def split_block(kind, lines, limit):
    """Return a block as one or more strings of at most about `limit` chars."""
    text = "\n".join(lines)
    if len(text) <= limit:
        return [text]
    if kind == "table" and len(lines) > 2 and set(lines[1].replace("|", "").strip()) <= set("-: "):
        return pack(lines[2:], limit, header=lines[:2])  # repeat the header row and separator
    if kind == "code" and len(lines) > 2:
        closed = FENCE.match(lines[-1]) is not None
        inner = lines[1:-1] if closed else lines[1:]
        return pack(inner, limit, header=lines[:1], footer=[FENCE.match(lines[0]).group(1)])
    return pack(lines, limit)


def split_section(lines, limit):
    """Join a section's blocks into pieces of at most `limit` chars."""
    text = "\n".join(lines).strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces, cur = [], ""
    for kind, block_lines in blocks(lines):
        for part in split_block(kind, block_lines, limit):
            if cur and len(cur) + len(part) + 2 > limit:
                pieces.append(cur)
                cur = ""
            cur = f"{cur}\n\n{part}" if cur else part
    if cur:
        pieces.append(cur)
    return pieces


def chunk_markdown(path, text, limit=MAX_CHARS):
    """Return (frontmatter, [(heading_trail, chunk_text), ...]) for one markdown file."""
    frontmatter, body = split_frontmatter(text)
    chunks = []
    for trail, lines in sections(body):
        heading = " > ".join(trail)
        prefix = f"{path} > {heading}" if heading else path
        for piece in split_section(lines, limit):
            chunks.append((heading, f"{prefix}\n\n{piece}"))
    return frontmatter, chunks
