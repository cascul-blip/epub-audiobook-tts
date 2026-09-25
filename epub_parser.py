"""Parse an epub into ordered chapters of plain-text paragraphs."""

import warnings
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

DEFAULT_SKIP_WORD_THRESHOLD = 100
TITLE_MAX_PARAGRAPH_WORDS = 12
TITLE_MAX_LEADING_PARAGRAPHS = 3

_STRIP_TAGS = ("script", "style", "svg", "nav")
_BLOCK_TAGS = ("p", "div", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "td")


@dataclass
class SpineDoc:
    index: int
    href: str
    paragraphs: list[str]
    word_count: int
    title: str


@dataclass
class Chapter:
    number: int
    spine_index: int
    title: str
    paragraphs: list[str]
    word_count: int


def word_count(text: str) -> int:
    return len(text.split())


def extract_paragraphs(xhtml_bytes: bytes) -> list[str]:
    soup = BeautifulSoup(xhtml_bytes, "lxml")

    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    body = soup.body or soup
    paragraphs = []
    for el in body.find_all(_BLOCK_TAGS):
        # Skip nested block elements already captured via an ancestor
        # (e.g. a <li> inside a <div> we already pulled text from).
        if el.find_parent(_BLOCK_TAGS) is not None:
            continue
        text = " ".join(el.get_text(separator=" ", strip=True).split())
        if not text:
            continue
        paragraphs.append(text)
    return paragraphs


def derive_title(paragraphs: list[str], fallback: str) -> str:
    pieces = []
    for p in paragraphs[:TITLE_MAX_LEADING_PARAGRAPHS]:
        if word_count(p) <= TITLE_MAX_PARAGRAPH_WORDS:
            pieces.append(p)
        else:
            break
    if not pieces:
        return fallback
    return " - ".join(pieces)


def load_spine_docs(epub_path: Path) -> list[SpineDoc]:
    book = epub.read_epub(str(epub_path))
    docs = []
    for index, (idref, _linear) in enumerate(book.spine):
        item = book.get_item_with_id(idref)
        if item is None:
            continue
        paragraphs = extract_paragraphs(item.get_content())
        wc = sum(word_count(p) for p in paragraphs)
        title = derive_title(paragraphs, fallback=f"Chapter {index + 1}")
        docs.append(SpineDoc(
            index=index,
            href=item.get_name(),
            paragraphs=paragraphs,
            word_count=wc,
            title=title,
        ))
    return docs


def build_chapters(
    spine_docs: list[SpineDoc],
    skip_word_threshold: int = DEFAULT_SKIP_WORD_THRESHOLD,
) -> list[Chapter]:
    chapters = []
    number = 0
    for doc in spine_docs:
        if doc.word_count < skip_word_threshold:
            continue
        number += 1
        chapters.append(Chapter(
            number=number,
            spine_index=doc.index,
            title=doc.title,
            paragraphs=doc.paragraphs,
            word_count=doc.word_count,
        ))
    return chapters
