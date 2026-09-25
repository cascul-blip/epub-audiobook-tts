"""Split chapter paragraphs into sentence-aware chunks safe for a single
ChatterboxTTS.generate() call (which has a hard ~40s / ~1000-speech-token
cap and silently truncates text that runs over it)."""

import re
from dataclasses import dataclass

DEFAULT_MAX_CHUNK_CHARS = 300

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc",
    "approx", "gen", "rev", "no", "vol", "fig", "co", "inc", "ltd",
}

_SENT_BOUNDARY_RE = re.compile(r'(?<=[.!?])[)"”’]?\s+(?=[A-Z0-9"“‘(])')
_TRAILING_WORD_RE = re.compile(r'([A-Za-z]+)\.$')


@dataclass
class Chunk:
    text: str
    paragraph_end: bool


def split_sentences(paragraph: str) -> list[str]:
    raw_pieces = _SENT_BOUNDARY_RE.split(paragraph.strip())
    sentences = []
    buffer = ""
    for piece in raw_pieces:
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        m = _TRAILING_WORD_RE.search(buffer)
        if m and m.group(1).lower() in _ABBREVIATIONS:
            # False split after an abbreviation (e.g. "Dr.") - keep merging.
            continue
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)
    return [s for s in sentences if s]


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]

    for sep in ("; ", ", "):
        if sep in sentence:
            clauses = sentence.split(sep)
            pieces = []
            buffer = ""
            for clause in clauses:
                candidate = f"{buffer}{sep}{clause}" if buffer else clause
                if len(candidate) <= max_chars:
                    buffer = candidate
                else:
                    if buffer:
                        pieces.extend(_split_long_sentence(buffer, max_chars))
                    buffer = clause
            if buffer:
                pieces.extend(_split_long_sentence(buffer, max_chars))
            return pieces

    # No punctuation to split on - hard-wrap at the last whitespace.
    pieces = []
    remaining = sentence
    while len(remaining) > max_chars:
        break_at = remaining.rfind(" ", 0, max_chars)
        if break_at <= 0:
            break_at = max_chars
        pieces.append(remaining[:break_at].strip())
        remaining = remaining[break_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_chapter(paragraphs: list[str], max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[Chunk]:
    chunks = []
    for paragraph in paragraphs:
        sentences = []
        for sentence in split_sentences(paragraph):
            sentences.extend(_split_long_sentence(sentence, max_chars))

        buffer = ""
        packed = []
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if buffer and len(candidate) > max_chars:
                packed.append(buffer)
                buffer = sentence
            else:
                buffer = candidate
        if buffer:
            packed.append(buffer)

        for i, text in enumerate(packed):
            chunks.append(Chunk(text=text, paragraph_end=(i == len(packed) - 1)))

    return chunks
