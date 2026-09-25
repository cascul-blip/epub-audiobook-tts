"""Model loading, per-chunk synthesis with a resume cache, and mp3 export."""

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment

from text_chunker import Chunk

DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5
DEFAULT_INTER_CHUNK_SILENCE_MS = 250
DEFAULT_PARAGRAPH_SILENCE_MS = 650
DEFAULT_MP3_BITRATE = "192k"
CACHE_DIRNAME = ".cache"
VOICES_DIR = Path(__file__).resolve().parent / "voices"


def resolve_voice(spec: str | None) -> Path | None:
    """Map a --voice value to a reference wav: a name in voices/ or a direct path.
    Returns None for the built-in default voice."""
    if not spec:
        return None
    named = VOICES_DIR / f"{spec}.wav"
    if named.exists():
        return named
    path = Path(spec).expanduser()
    if path.is_file():
        return path
    available = sorted(v.stem for v in VOICES_DIR.glob("*.wav") if not v.stem.endswith("_preview"))
    raise SystemExit(
        f"Unknown voice {spec!r}: not in {VOICES_DIR}/ and not an audio file path. "
        f"Available voices: {available or 'none (create one with prepare_voice.py)'}"
    )


def load_model(device: str = "cuda", voice_path: Path | None = None,
               exaggeration: float = DEFAULT_EXAGGERATION):
    from chatterbox.tts import ChatterboxTTS  # deferred: keep --list torch-free
    model = ChatterboxTTS.from_pretrained(device=device)
    if voice_path is not None:
        # generate() reuses model.conds, so cloning once here applies to every chunk.
        model.prepare_conditionals(str(voice_path), exaggeration=exaggeration)
    return model


def synth_chunk(
    model,
    text: str,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
) -> np.ndarray:
    wav = model.generate(text=text, exaggeration=exaggeration, cfg_weight=cfg_weight)
    return wav.squeeze().cpu().numpy().astype(np.float32)


def chunk_cache_path(cache_root: Path, chapter_number: int, chunk_index: int) -> Path:
    return cache_root / f"ch{chapter_number:02d}" / f"chunk_{chunk_index:04d}.wav"


def get_or_synthesize_chunk(
    model,
    text: str,
    cache_path: Path,
    sr: int,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    force: bool = False,
) -> tuple[np.ndarray, bool]:
    """Returns (audio, from_cache)."""
    if not force and cache_path.exists():
        try:
            wav, file_sr = sf.read(cache_path, dtype="float32")
            if file_sr == sr:
                return wav, True
        except Exception:
            pass  # fall through to re-synthesize

    wav = synth_chunk(model, text, exaggeration=exaggeration, cfg_weight=cfg_weight)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # avoid allocator fragmentation over many small calls
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(cache_path, wav, sr)
    return wav, False


def build_silence(duration_ms: int, sr: int) -> np.ndarray:
    return np.zeros(int(sr * duration_ms / 1000), dtype=np.float32)


def assemble_chapter_audio(
    model,
    chapter,
    chunks: list[Chunk],
    cache_root: Path,
    total_chapters: int,
    force: bool = False,
    exaggeration: float = DEFAULT_EXAGGERATION,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    inter_chunk_silence_ms: int = DEFAULT_INTER_CHUNK_SILENCE_MS,
    paragraph_silence_ms: int = DEFAULT_PARAGRAPH_SILENCE_MS,
) -> np.ndarray:
    sr = model.sr
    pieces = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        cache_path = chunk_cache_path(cache_root, chapter.number, i)
        wav, from_cache = get_or_synthesize_chunk(
            model, chunk.text, cache_path, sr,
            exaggeration=exaggeration, cfg_weight=cfg_weight, force=force,
        )
        tag = "cached" if from_cache else "synthesized"
        print(f"  chapter {chapter.number}/{total_chapters}, chunk {i + 1}/{total} [{tag}]")
        pieces.append(wav)
        if i < total - 1:
            silence_ms = paragraph_silence_ms if chunk.paragraph_end else inter_chunk_silence_ms
            pieces.append(build_silence(silence_ms, sr))

    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def export_mp3(wav: np.ndarray, sr: int, out_path: Path, bitrate: str = DEFAULT_MP3_BITRATE) -> None:
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    segment = AudioSegment(data=pcm16, sample_width=2, frame_rate=sr, channels=1)

    part_path = out_path.with_suffix(out_path.suffix + ".part")
    segment.export(part_path, format="mp3", bitrate=bitrate)
    os.replace(part_path, out_path)
