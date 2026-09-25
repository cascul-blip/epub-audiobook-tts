#!/usr/bin/env python3
"""Turn a video or audio clip into a reference wav in voices/ for Chatterbox voice cloning."""

import argparse
import shutil
import subprocess
from pathlib import Path

import soundfile as sf

import tts_synth

REFERENCE_SR = 24000  # Chatterbox's S3GEN_SR; the model resamples anyway, this avoids a lossy extra step
DEFAULT_DURATION_S = 20.0
MIN_GOOD_DURATION_S = 5.0
MAX_GOOD_DURATION_S = 30.0
DEFAULT_DEVICE = "cuda"

# Trim silence from both ends (silenceremove only handles the start, hence the areverse
# sandwich), cut rumble, and normalize loudness so quiet clips clone as well as loud ones.
_TRIM_SILENCE = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1"
AUDIO_FILTER = ",".join([
    "highpass=f=80",
    _TRIM_SILENCE, "areverse", _TRIM_SILENCE, "areverse",
    "loudnorm=I=-20:TP=-2",
])


def extract_reference(input_path: Path, out_path: Path, start: str | None, duration: float) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", start]
    cmd += [
        "-t", str(duration), "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", str(REFERENCE_SR),
        "-af", AUDIO_FILTER,
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{result.stderr.strip()}")


def list_voices() -> None:
    voices = sorted(v for v in tts_synth.VOICES_DIR.glob("*.wav") if not v.stem.endswith("_preview"))
    if not voices:
        print(f"No voices yet in {tts_synth.VOICES_DIR}/")
        return
    for v in voices:
        print(f"{v.stem:<24} {sf.info(v).duration:5.1f}s")


def write_preview(voice_path: Path, text: str, device: str, exaggeration: float, cfg_weight: float) -> Path:
    model = tts_synth.load_model(device=device, voice_path=voice_path, exaggeration=exaggeration)
    wav = tts_synth.synth_chunk(model, text, exaggeration=exaggeration, cfg_weight=cfg_weight)
    preview_path = voice_path.with_name(f"{voice_path.stem}_preview.wav")
    sf.write(preview_path, wav, model.sr)
    return preview_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare_voice.py",
        description="Extract a clean speech reference from a video/audio clip into voices/<name>.wav.",
    )
    p.add_argument("input_path", type=Path, nargs="?", help="Any video or audio file ffmpeg can read.")
    p.add_argument("name", nargs="?", help="Voice name, used later as: epub_to_audiobook.py --voice NAME")
    p.add_argument(
        "--start", default=None,
        help="Where the clean speech begins, e.g. '83', '1:23' or '00:01:23.5'. Default: start of clip.",
    )
    p.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S,
        help=f"Seconds to take from --start (default {DEFAULT_DURATION_S:g}). "
             "The first ~10s of the result matter most.",
    )
    p.add_argument(
        "--preview", metavar="TEXT", default=None,
        help="Also synthesize TEXT in the cloned voice to voices/<name>_preview.wav (loads the model).",
    )
    p.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda", "cpu", "mps"])
    p.add_argument("--exaggeration", type=float, default=tts_synth.DEFAULT_EXAGGERATION)
    p.add_argument("--cfg-weight", type=float, default=tts_synth.DEFAULT_CFG_WEIGHT)
    p.add_argument("--list", action="store_true", help="List existing voices and exit.")
    return p


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_voices()
        return
    if args.input_path is None or args.name is None:
        parser.error("input_path and name are required (or use --list)")
    if not args.input_path.exists():
        raise SystemExit(f"No such file: {args.input_path}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    tts_synth.VOICES_DIR.mkdir(exist_ok=True)
    out_path = tts_synth.VOICES_DIR / f"{args.name}.wav"
    extract_reference(args.input_path, out_path, args.start, args.duration)

    duration = sf.info(out_path).duration
    print(f"[voice] wrote {out_path} ({duration:.1f}s of audio)")
    if duration < MIN_GOOD_DURATION_S:
        print(f"[warn] under {MIN_GOOD_DURATION_S:g}s: the clone may be weak. "
              "Pick a longer stretch of continuous speech (--start/--duration).")
    elif duration > MAX_GOOD_DURATION_S:
        print("[warn] long reference: only the first ~10s shape the voice, so make sure "
              "--start lands on clean speech.")

    if args.preview:
        preview_path = write_preview(out_path, args.preview, args.device, args.exaggeration, args.cfg_weight)
        print(f"[preview] wrote {preview_path}")

    print(f"Use it with: ./bin/python epub_to_audiobook.py BOOK.epub --voice {args.name}")


if __name__ == "__main__":
    main()
