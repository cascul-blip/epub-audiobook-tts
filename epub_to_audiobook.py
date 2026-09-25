#!/usr/bin/env python3
"""Convert an EPUB's chapters into per-chapter MP3s via Chatterbox TTS."""

import argparse
import re
from pathlib import Path

import epub_parser
import text_chunker
import tts_synth

DEFAULT_DEVICE = "cuda"
_ILLEGAL_FILENAME_CHARS_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(name: str, max_len: int = 80) -> str:
    cleaned = _ILLEGAL_FILENAME_CHARS_RE.sub("", name)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_len].rstrip()


def chapter_output_path(output_dir: Path, chapter: epub_parser.Chapter) -> Path:
    return output_dir / f"{chapter.number:02d} - {sanitize_filename(chapter.title)}.mp3"


def parse_chapter_selection(spec: str, available: list[int]) -> set[int]:
    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start, end = int(start_s), int(end_s)
            except ValueError:
                raise SystemExit(f"Invalid chapter range: {part!r}")
            selected.update(range(start, end + 1))
        else:
            try:
                selected.add(int(part))
            except ValueError:
                raise SystemExit(f"Invalid chapter number: {part!r}")

    invalid = selected - set(available)
    if invalid:
        raise SystemExit(
            f"Unknown chapter number(s): {sorted(invalid)}. "
            f"Available: {available}"
        )
    return selected


def print_chapter_table(spine_docs: list[epub_parser.SpineDoc], chapters: list[epub_parser.Chapter]) -> None:
    kept_by_index = {c.spine_index: c.number for c in chapters}
    print(f"{'#':>3}  {'words':>7}  {'status':<6}  {'ch#':>3}  title")
    for doc in spine_docs:
        ch_num = kept_by_index.get(doc.index)
        status = "keep" if ch_num is not None else "skip"
        ch_display = str(ch_num) if ch_num is not None else "-"
        title = doc.title if ch_num is not None else f"({doc.href})"
        print(f"{doc.index:>3}  {doc.word_count:>7}  {status:<6}  {ch_display:>3}  {title}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="epub_to_audiobook.py",
        description="Convert an EPUB's chapters to per-chapter MP3s via Chatterbox TTS.",
    )
    p.add_argument("epub_path", type=Path)
    p.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Default: '<epub-stem>_audiobook/' next to the epub.",
    )
    p.add_argument("--max-chunk-chars", type=int, default=text_chunker.DEFAULT_MAX_CHUNK_CHARS)
    p.add_argument("--skip-word-threshold", type=int, default=epub_parser.DEFAULT_SKIP_WORD_THRESHOLD)
    p.add_argument(
        "--list", action="store_true",
        help="Print detected chapters with word counts and skip/keep decisions. "
             "Does NOT load the TTS model or touch the GPU.",
    )
    p.add_argument(
        "--chapters", type=str, default=None,
        help="Comma-separated chapter numbers/ranges from --list output, e.g. '2,4-6'. "
             "Default: all kept chapters.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ignore cached chunk audio and existing chapter mp3s; regenerate everything.",
    )
    p.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda", "cpu", "mps"])
    p.add_argument("--exaggeration", type=float, default=tts_synth.DEFAULT_EXAGGERATION)
    p.add_argument(
        "--voice", default=None,
        help="Clone a voice: a name from voices/ (see prepare_voice.py --list) or a path "
             "to a reference audio file. Default: Chatterbox's built-in voice.",
    )
    p.add_argument(
        "--cfg-weight", type=float, default=tts_synth.DEFAULT_CFG_WEIGHT,
        help="Pacing/guidance weight. ~0.3 often suits cloned or fast-talking voices.",
    )
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    if not args.epub_path.exists():
        raise SystemExit(f"No such file: {args.epub_path}")

    spine_docs = epub_parser.load_spine_docs(args.epub_path)
    chapters = epub_parser.build_chapters(spine_docs, skip_word_threshold=args.skip_word_threshold)

    if not chapters:
        raise SystemExit(
            "No chapters detected above --skip-word-threshold "
            f"({args.skip_word_threshold}). Try a lower threshold or check --list output."
        )

    if args.list:
        print_chapter_table(spine_docs, chapters)
        return

    voice_path = tts_synth.resolve_voice(args.voice)  # fail fast, before touching the GPU
    voice_name = voice_path.stem if voice_path else None

    default_dirname = args.epub_path.stem + "_audiobook" + (f"_{voice_name}" if voice_name else "")
    output_dir = args.output_dir or args.epub_path.with_name(default_dirname)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = None
    if args.chapters:
        selected = parse_chapter_selection(args.chapters, [c.number for c in chapters])

    to_process = []
    for chapter in chapters:
        if selected is not None and chapter.number not in selected:
            continue
        out_path = chapter_output_path(output_dir, chapter)
        if out_path.exists() and not args.force:
            print(f"[skip] chapter {chapter.number}: {out_path.name} already exists")
            continue
        to_process.append((chapter, out_path))

    if not to_process:
        return

    if voice_path:
        print(f"[voice] cloning from {voice_path}")
    model = tts_synth.load_model(
        device=args.device, voice_path=voice_path, exaggeration=args.exaggeration,
    )
    # Per-voice cache so chunks from different voices never get mixed.
    cache_root = output_dir / tts_synth.CACHE_DIRNAME / (voice_name or "default")

    for chapter, out_path in to_process:
        print(f"[start] chapter {chapter.number}: {chapter.title} ({chapter.word_count} words)")
        chunks = text_chunker.chunk_chapter(chapter.paragraphs, max_chars=args.max_chunk_chars)
        wav = tts_synth.assemble_chapter_audio(
            model, chapter, chunks, cache_root, total_chapters=len(chapters),
            force=args.force, exaggeration=args.exaggeration, cfg_weight=args.cfg_weight,
        )
        tts_synth.export_mp3(wav, model.sr, out_path)
        print(f"[done] chapter {chapter.number}: {out_path.name}")


if __name__ == "__main__":
    main()
