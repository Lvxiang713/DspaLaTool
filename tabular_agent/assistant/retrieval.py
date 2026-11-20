"""Deterministic keyword retrieval over the bundled knowledge base and skills."""
import re

_TOKEN = re.compile(r"[a-z0-9_]+")


def _load(group):
    base = __package__ + "." + group
    try:
        from importlib.resources import files
        names = sorted(p.name for p in files(base).iterdir() if p.name.endswith(".md"))
        read = lambda n: files(base).joinpath(n).read_text(encoding="utf-8")
    except Exception:
        # Fallback for zip-less editable layouts without importlib.resources metadata.
        from pathlib import Path
        root = Path(__file__).resolve().parent / group
        names = sorted(p.name for p in root.glob("*.md"))
        read = lambda n: (root / n).read_text(encoding="utf-8")
    docs = []
    for name in names:
        text = read(name)
        meta, body = _split_front_matter(text)
        docs.append({
            "id": meta.get("id", name[:-3]),
            "kind": group,
            "title": meta.get("title", name[:-3].replace("_", " ").title()),
            "keywords": [w for w in meta.get("keywords", "").split() if w],
            "body": body.strip(),
        })
    return docs


def _split_front_matter(text):
    meta = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip()
            body = parts[2]
    return meta, body


def _tokens(text):
    stop = {"the", "a", "an", "is", "are", "how", "do", "i", "to", "of", "in",
            "and", "or", "can", "my", "this", "it", "with", "what", "for"}
    return [w for w in _TOKEN.findall(text.lower()) if w not in stop]


def _score(query_tokens, query_lower, doc):
    haystack = set(_tokens(doc["title"] + " " + doc["body"]))
    keywords = set(k.lower() for k in doc["keywords"])
    score = 0.0
    for token in set(query_tokens):
        if token in keywords:
            score += 3.0
        elif token in haystack:
            score += 1.0
    return score


def retrieve(question, limit=3, kinds=("knowledge", "skills")):
    """Return ranked document dicts for a question. limit=0 returns nothing."""
    if limit <= 0:
        return []
    query_lower = question.lower()
    query_tokens = _tokens(question)
    docs = []
    for kind in kinds:
        docs.extend(_load(kind))
    scored = [(_score(query_tokens, query_lower, d), d) for d in docs]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: (-item[0], item[1]["kind"] != "knowledge", item[1]["id"]))
    return [d for _, d in scored[:limit]]
