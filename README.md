# epub-audiobook-tts

Convert EPUB books into per-chapter MP3 audiobooks with Chatterbox TTS, with optional voice cloning.

Setup (needs Python 3.14 and ffmpeg on PATH; the project folder doubles as the venv):
python -m venv .
./bin/pip install -r requirements.txt

- epub_parser.py — parses the epub spine into chapters, auto-skipping front matter (cover/TOC) via word count, deriving titles from leading short paragraphs.
- text_chunker.py — splits chapter text into sentence-aware chunks (≤300 chars) safe for Chatterbox's hard ~40s/generate-call limit.
- tts_synth.py — loads the model once, synthesizes/caches each chunk as a wav, concatenates with paragraph-aware pauses, exports mp3.
- epub_to_audiobook.py — the CLI.
- prepare_voice.py — turns a video/audio clip into a voices/<name>.wav reference for voice cloning.

Usage:
./bin/python epub_to_audiobook.py "book.epub" --list                    # preview chapters, no GPU
./bin/python epub_to_audiobook.py "book.epub"                            # convert everything
./bin/python epub_to_audiobook.py "book.epub" --chapters 2,4-6           # subset
./bin/python epub_to_audiobook.py "book.epub" --force                    # ignore cache/existing output
./bin/python epub_to_audiobook.py "book.epub" --voice NAME               # use a cloned voice

Voice cloning:
./bin/python prepare_voice.py clip.mp4 NAME --start 1:23 --duration 20 --preview "Testing the cloned voice."
./bin/python prepare_voice.py --list                                     # show saved voices
- Works with any video or audio file ffmpeg can read. It extracts mono 24kHz audio, trims silence and normalizes loudness.
- Choose a stretch with one person speaking and no music. Only the first ~10s of the reference shape the voice.
- Listen to voices/NAME_preview.wav, and adjust --start if the clone sounds off.
- --voice also accepts a direct path to an audio file.
- Cloned output goes to '<epub-stem>_audiobook_NAME/', and each voice gets its own chunk cache (.cache/NAME/).
- If a clone sounds rushed, try --cfg-weight 0.3. Raise --exaggeration for more expressive delivery.
