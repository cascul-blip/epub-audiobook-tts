#!/usr/bin/env python3
"""Qt front-end for epub_to_audiobook.py: pick a book, set the flags, tick chapters, preview, run."""

import html
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFontDatabase, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

import epub_parser
import epub_to_audiobook
import text_chunker
import tts_synth

CLI_SCRIPT = Path(epub_to_audiobook.__file__).resolve()
DEFAULT_PREVIEW_PARAGRAPHS = 3
PREVIEW_MAX_PARAGRAPH_CHARS = 400
STOP_KILL_TIMEOUT_MS = 3000
LOG_MAX_LINES = 5000

_PARSER = epub_to_audiobook.build_arg_parser()
_ACTIONS = {a.dest: a for a in _PARSER._actions}
_PROGRESS_RE = re.compile(r"chapter (\d+)/\d+, chunk (\d+)/(\d+) \[(\w+)\]")

COL_CHECK, COL_SPINE, COL_CHAPTER, COL_WORDS, COL_STATUS, COL_TITLE = range(6)
_HEADERS = ["", "#", "ch#", "words", "status", "title"]


def cli_default(dest: str):
    return _PARSER.get_default(dest)


def cli_tooltip(dest: str) -> str:
    action = _ACTIONS[dest]
    flag = action.option_strings[-1] if action.option_strings else dest
    return f"{flag}\n{action.help}" if action.help else flag


def format_duration(seconds: float) -> str:
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes}m {secs:02d}s"


def compress_ranges(numbers) -> str:
    """[1, 3, 4, 5, 8] -> '1,3-5,8' (the inverse of epub_to_audiobook.parse_chapter_selection)."""
    parts = []
    run_start = prev = None
    for n in sorted(set(numbers)):
        if prev is not None and n == prev + 1:
            prev = n
            continue
        if prev is not None:
            parts.append(str(run_start) if run_start == prev else f"{run_start}-{prev}")
        run_start = prev = n
    if prev is not None:
        parts.append(str(run_start) if run_start == prev else f"{run_start}-{prev}")
    return ",".join(parts)


def default_output_dir(epub_path: Path, voice: str) -> Path:
    """Mirrors the default in epub_to_audiobook.main()."""
    suffix = f"_{Path(voice).stem}" if voice else ""
    return epub_path.with_name(f"{epub_path.stem}_audiobook{suffix}")


@dataclass
class FormState:
    epub_path: str
    output_dir: str = ""
    voice: str = ""  # "" = Chatterbox's built-in voice
    device: str = cli_default("device")
    exaggeration: float = cli_default("exaggeration")
    cfg_weight: float = cli_default("cfg_weight")
    max_chunk_chars: int = cli_default("max_chunk_chars")
    skip_word_threshold: int = cli_default("skip_word_threshold")
    force: bool = False
    chapters: str | None = None  # None = all kept chapters


def build_cli_args(state: FormState) -> list[str]:
    """Arguments for epub_to_audiobook.py; flags left at their default are omitted."""
    args = [state.epub_path]
    if state.output_dir:
        args += ["--output-dir", state.output_dir]
    if state.voice:
        args += ["--voice", state.voice]
    if state.device != cli_default("device"):
        args += ["--device", state.device]
    for dest in ("exaggeration", "cfg_weight"):
        value = getattr(state, dest)
        if abs(value - cli_default(dest)) > 1e-9:
            args += [_ACTIONS[dest].option_strings[-1], f"{value:g}"]
    for dest in ("max_chunk_chars", "skip_word_threshold"):
        value = getattr(state, dest)
        if value != cli_default(dest):
            args += [_ACTIONS[dest].option_strings[-1], str(value)]
    if state.chapters:
        args += ["--chapters", state.chapters]
    if state.force:
        args.append("--force")
    return args


class EpubLoader(QThread):
    loaded = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self) -> None:
        try:
            docs = epub_parser.load_spine_docs(Path(self.path))
        except Exception as e:
            self.failed.emit(self.path, f"{type(e).__name__}: {e}")
        else:
            self.loaded.emit(self.path, docs)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EPUB to Audiobook")
        self.resize(1200, 850)
        self.setAcceptDrops(True)

        self.epub_path = ""  # the book that is loaded (or loading)
        self.spine_docs: list[epub_parser.SpineDoc] | None = None
        self.chapters: list[epub_parser.Chapter] = []
        self.unchecked: set[int] = set()  # spine indices, so ticks survive renumbering
        self.search_hits: list[tuple[int, int]] = []  # (table row, paragraph index)
        self.search_pos = -1
        self.process: QProcess | None = None
        self._out_buffer = ""
        self._planned = 0
        self._started = 0
        self._run_started_at = 0.0     # Start pressed (includes model loading)
        self._synth_started_at = None  # first chapter began; the ETA clock
        self._synth_count = 0
        self._eta_at = None            # monotonic time the run is expected to end
        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(1000)
        self._eta_timer.timeout.connect(self._update_eta_text)

        self._build_ui()
        self._restore_settings()
        self._refresh()

    # ---- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        self.book_box = QGroupBox("Book")
        grid = QGridLayout(self.book_box)
        self.epub_edit = QLineEdit()
        self.epub_edit.setPlaceholderText("Path to an .epub (or drop one on the window)")
        self.epub_edit.editingFinished.connect(lambda: self.set_epub(self.epub_edit.text()))
        epub_browse = QPushButton("Browse…")
        epub_browse.clicked.connect(self._browse_epub)
        self.output_edit = QLineEdit()
        self.output_edit.setToolTip(cli_tooltip("output_dir"))
        self.output_edit.textChanged.connect(self._refresh)
        output_browse = QPushButton("Browse…")
        output_browse.clicked.connect(self._browse_output)
        grid.addWidget(QLabel("EPUB"), 0, 0)
        grid.addWidget(self.epub_edit, 0, 1)
        grid.addWidget(epub_browse, 0, 2)
        grid.addWidget(QLabel("Output dir"), 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(output_browse, 1, 2)

        self.options_box = QGroupBox("Options")
        form = QFormLayout(self.options_box)

        self.voice_combo = QComboBox()
        self.voice_combo.setToolTip(cli_tooltip("voice"))
        self.voice_combo.addItem("Built-in default", "")
        for voice in tts_synth.list_voice_paths():
            self.voice_combo.addItem(voice.stem, voice.stem)
        self.voice_combo.currentIndexChanged.connect(self._refresh)
        voice_file = QPushButton("Audio file…")
        voice_file.clicked.connect(self._browse_voice)
        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice_combo, 1)
        voice_row.addWidget(voice_file)
        form.addRow("Voice", voice_row)

        self.device_combo = QComboBox()
        self.device_combo.setToolTip(cli_tooltip("device"))
        self.device_combo.addItems(_ACTIONS["device"].choices)
        self.device_combo.setCurrentText(cli_default("device"))
        self.device_combo.currentIndexChanged.connect(self._refresh)
        form.addRow("Device", self.device_combo)

        self.exaggeration_spin = self._float_spin("exaggeration", maximum=2.0)
        form.addRow("Exaggeration", self.exaggeration_spin)
        self.cfg_spin = self._float_spin("cfg_weight", maximum=1.0)
        form.addRow("CFG weight", self.cfg_spin)

        self.chunk_spin = self._int_spin("max_chunk_chars", minimum=50, maximum=2000)
        self.chunk_spin.valueChanged.connect(self._refresh)
        form.addRow("Max chunk chars", self.chunk_spin)
        self.threshold_spin = self._int_spin("skip_word_threshold", minimum=0, maximum=1_000_000)
        self.threshold_spin.valueChanged.connect(self._rebuild_chapters)
        form.addRow("Skip word threshold", self.threshold_spin)

        self.force_check = QCheckBox("Force regenerate")
        self.force_check.setToolTip(cli_tooltip("force"))
        self.force_check.toggled.connect(self._refresh)
        form.addRow(self.force_check)

        self.preview_spin = QSpinBox()
        self.preview_spin.setRange(1, 20)
        self.preview_spin.setValue(DEFAULT_PREVIEW_PARAGRAPHS)
        self.preview_spin.setToolTip("Paragraphs shown per chapter in the preview (GUI only).")
        self.preview_spin.valueChanged.connect(self._refresh_preview)
        form.addRow("Preview paragraphs", self.preview_spin)

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.setToolTip(cli_tooltip("chapters"))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_TITLE, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.currentCellChanged.connect(self._on_row_selected)

        self.selection_label = QLabel()
        select_row = QHBoxLayout()
        for label, slot in (("All", self._select_all), ("None", self._select_none), ("Invert", self._select_invert)):
            button = QPushButton(label)
            button.clicked.connect(slot)
            select_row.addWidget(button)
        select_row.addWidget(self.selection_label, 1)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Find text in the book (Ctrl+F), Enter for the next match")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._run_search)
        self.search_edit.returnPressed.connect(lambda: self._step_search(1))
        search_prev = QPushButton("Previous")
        search_prev.clicked.connect(lambda: self._step_search(-1))
        search_next = QPushButton("Next")
        search_next.clicked.connect(lambda: self._step_search(1))
        self.search_label = QLabel()
        search_row = QHBoxLayout()
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(search_prev)
        search_row.addWidget(search_next)
        search_row.addWidget(self.search_label)
        QShortcut(QKeySequence.StandardKey.Find, self, self._focus_search)

        chapters_box = QGroupBox("Chapters")
        chapters_layout = QVBoxLayout(chapters_box)
        chapters_layout.addLayout(search_row)
        chapters_layout.addWidget(self.table)
        chapters_layout.addLayout(select_row)

        self.preview = QTextBrowser()
        self.preview_box = preview_box = QGroupBox()
        QVBoxLayout(preview_box).addWidget(self.preview)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(chapters_box)
        right.addWidget(preview_box)
        right.setSizes([320, 280])

        middle = QSplitter(Qt.Orientation.Horizontal)
        middle.addWidget(self.options_box)
        middle.addWidget(right)
        middle.setStretchFactor(1, 1)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.book_box)
        top_layout.addWidget(middle, 1)

        self.command_edit = QLineEdit()
        self.command_edit.setReadOnly(True)
        self.command_edit.setToolTip("The equivalent command line.")
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress_label = QLabel()
        run_row = QHBoxLayout()
        run_row.addWidget(self.start_button)
        run_row.addWidget(self.stop_button)
        run_row.addWidget(self.progress, 1)
        run_row.addWidget(self.progress_label)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(LOG_MAX_LINES)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self.command_edit)
        bottom_layout.addLayout(run_row)
        bottom_layout.addWidget(self.log, 1)

        root = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(top)
        root.addWidget(bottom)
        root.setSizes([680, 170])
        root.setContentsMargins(8, 8, 8, 8)
        self.setCentralWidget(root)

    def _float_spin(self, dest: str, maximum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(cli_default(dest))
        spin.setToolTip(cli_tooltip(dest))
        spin.valueChanged.connect(self._refresh)
        return spin

    def _int_spin(self, dest: str, minimum: int, maximum: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(cli_default(dest))
        spin.setToolTip(cli_tooltip(dest))
        spin.setKeyboardTracking(False)  # don't re-filter chapters on every keystroke
        return spin

    # ---- Book loading ----------------------------------------------------

    def set_epub(self, path: str) -> None:
        path = path.strip()
        if path:
            path = str(Path(path).expanduser().resolve())
        if path == self.epub_path:
            return
        self.epub_path = path
        self.epub_edit.setText(path)
        self.spine_docs = None
        self.unchecked.clear()
        self._run_search()
        self._rebuild_chapters()
        if not path:
            return
        if not Path(path).is_file():
            self.preview.setPlainText(f"No such file: {path}")
            return
        self.preview.setPlainText("Loading…")
        loader = EpubLoader(path, self)
        loader.loaded.connect(self._on_loaded)
        loader.failed.connect(self._on_load_failed)
        loader.finished.connect(loader.deleteLater)
        loader.start()

    def _on_loaded(self, path: str, docs) -> None:
        if path != self.epub_path:
            return  # a newer book was picked while this one was parsing
        self.spine_docs = docs
        self._rebuild_chapters()
        self._run_search()

    def _on_load_failed(self, path: str, message: str) -> None:
        if path == self.epub_path:
            self.preview.setPlainText(f"Could not read {path}\n{message}")

    def _rebuild_chapters(self) -> None:
        docs = self.spine_docs or []
        self.chapters = epub_parser.build_chapters(docs, skip_word_threshold=self.threshold_spin.value())
        by_spine = {c.spine_index: c for c in self.chapters}
        enabled = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

        self.table.blockSignals(True)
        self.table.setRowCount(len(docs))
        for row, doc in enumerate(docs):
            chapter = by_spine.get(doc.index)
            flags = enabled if chapter else Qt.ItemFlag.NoItemFlags
            check = QTableWidgetItem()
            check.setData(Qt.ItemDataRole.UserRole, doc.index)
            if chapter:
                check.setFlags(flags | Qt.ItemFlag.ItemIsUserCheckable)
                check.setCheckState(
                    Qt.CheckState.Unchecked if doc.index in self.unchecked else Qt.CheckState.Checked
                )
            else:
                check.setFlags(flags)
            self.table.setItem(row, COL_CHECK, check)
            cells = {
                COL_SPINE: str(doc.index),
                COL_CHAPTER: str(chapter.number) if chapter else "-",
                COL_WORDS: f"{doc.word_count:,}",
                COL_STATUS: "" if chapter else "skip",
                COL_TITLE: chapter.title if chapter else f"({doc.href})",
            }
            for col, text in cells.items():
                item = QTableWidgetItem(text)
                item.setFlags(flags)
                if col in (COL_SPINE, COL_CHAPTER, COL_WORDS):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self._refresh()

    # ---- Chapter selection -----------------------------------------------

    def selected_chapters(self) -> list[epub_parser.Chapter]:
        return [c for c in self.chapters if c.spine_index not in self.unchecked]

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != COL_CHECK:
            return
        spine_index = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.unchecked.discard(spine_index)
        else:
            self.unchecked.add(spine_index)
        self._refresh()

    def _set_unchecked(self, unchecked: set[int]) -> None:
        self.unchecked = unchecked
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_CHECK)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                spine_index = item.data(Qt.ItemDataRole.UserRole)
                item.setCheckState(
                    Qt.CheckState.Unchecked if spine_index in unchecked else Qt.CheckState.Checked
                )
        self.table.blockSignals(False)
        self._refresh()

    def _select_all(self) -> None:
        self._set_unchecked(set())

    def _select_none(self) -> None:
        self._set_unchecked({c.spine_index for c in self.chapters})

    def _select_invert(self) -> None:
        self._set_unchecked({c.spine_index for c in self.chapters} - self.unchecked)

    def _on_row_selected(self, row: int, *_unused) -> None:
        item = self.table.item(row, COL_CHAPTER) if row >= 0 else None
        if item is not None and item.text().isdigit():
            self.preview.scrollToAnchor(f"ch{item.text()}")

    # ---- Text search -----------------------------------------------------

    def _search_pattern(self) -> re.Pattern | None:
        needle = self.search_edit.text().strip()
        return re.compile(re.escape(needle), re.IGNORECASE) if needle else None

    def _focus_search(self) -> None:
        self.search_edit.setFocus()
        self.search_edit.selectAll()

    def _run_search(self, *_unused) -> None:
        """Collect every paragraph containing the search text and jump to the first one."""
        pattern = self._search_pattern()
        self.search_hits = []
        if pattern is not None:
            # Table rows are in spine order, so the row is the doc's position in spine_docs.
            self.search_hits = [
                (row, i)
                for row, doc in enumerate(self.spine_docs or [])
                for i, paragraph in enumerate(doc.paragraphs)
                if pattern.search(paragraph)
            ]
        self.search_pos = 0 if self.search_hits else -1
        self._show_search_hit()

    def _step_search(self, step: int) -> None:
        if self.search_hits:
            self.search_pos = (self.search_pos + step) % len(self.search_hits)
            self._show_search_hit()

    def _show_search_hit(self) -> None:
        if self.search_hits:
            row, _ = self.search_hits[self.search_pos]
            sections = len({r for r, _ in self.search_hits})
            self.search_label.setText(
                f"{self.search_pos + 1} of {len(self.search_hits)} in {sections} section(s)"
            )
            self.table.setCurrentCell(row, COL_TITLE)
            self.table.scrollToItem(self.table.item(row, COL_TITLE))
        else:
            searching = self._search_pattern() is not None and self.spine_docs is not None
            self.search_label.setText("no matches" if searching else "")
        self._refresh_preview()

    def _search_hit_html(self) -> str:
        """The matching paragraph with its neighbours, matches highlighted."""
        pattern = self._search_pattern()
        row, index = self.search_hits[self.search_pos]
        doc = self.spine_docs[row]
        number = self.table.item(row, COL_CHAPTER).text()
        where = f"{number} · {doc.title}" if number.isdigit() else f"Skipped section ({doc.href})"
        parts = [
            f'<h3>{html.escape(where)} <span style="color: gray; font-weight: normal">'
            f"(paragraph {index + 1} of {len(doc.paragraphs)})</span></h3>"
        ]
        for i in range(max(0, index - 1), min(len(doc.paragraphs), index + 2)):
            pieces = []
            position = 0
            for match in pattern.finditer(doc.paragraphs[i]):
                pieces.append(html.escape(doc.paragraphs[i][position:match.start()]))
                if i == index and position == 0:
                    pieces.append('<a name="hit"></a>')  # scroll target: the first match itself
                pieces.append(
                    f'<span style="background-color: #ffd54a; color: black">{html.escape(match.group())}</span>'
                )
                position = match.end()
            pieces.append(html.escape(doc.paragraphs[i][position:]))
            parts.append(f"<p>{''.join(pieces)}</p>")
        return "".join(parts)

    # ---- Derived state ---------------------------------------------------

    def form_state(self) -> FormState:
        selected = self.selected_chapters()
        all_selected = len(selected) == len(self.chapters)
        return FormState(
            epub_path=self.epub_path,
            output_dir=self.output_edit.text().strip(),
            voice=self.voice_combo.currentData(),
            device=self.device_combo.currentText(),
            exaggeration=self.exaggeration_spin.value(),
            cfg_weight=self.cfg_spin.value(),
            max_chunk_chars=self.chunk_spin.value(),
            skip_word_threshold=self.threshold_spin.value(),
            force=self.force_check.isChecked(),
            chapters=None if all_selected else compress_ranges(c.number for c in selected),
        )

    def output_dir(self) -> Path | None:
        explicit = self.output_edit.text().strip()
        if explicit:
            return Path(explicit).expanduser()
        if not self.epub_path:
            return None
        return default_output_dir(Path(self.epub_path), self.voice_combo.currentData())

    def _refresh(self, *_unused) -> None:
        default_dir = default_output_dir(Path(self.epub_path), self.voice_combo.currentData()) if self.epub_path else ""
        self.output_edit.setPlaceholderText(f"Default: {default_dir}" if default_dir else "")

        selected = self.selected_chapters()
        words = sum(c.word_count for c in selected)
        self.selection_label.setText(
            f"{len(selected)} of {len(self.chapters)} chapters selected · {words:,} words" if self.chapters else ""
        )
        self.command_edit.setText(
            shlex.join(["./bin/python", CLI_SCRIPT.name, *build_cli_args(self.form_state())])
            if self.epub_path else ""
        )
        self.command_edit.setCursorPosition(0)
        self.start_button.setEnabled(bool(selected) and self.process is None)
        self._refresh_status()
        self._refresh_preview()

    def _done_chapter_numbers(self) -> set[int]:
        output_dir = self.output_dir()
        if output_dir is None:
            return set()
        return {
            c.number for c in self.chapters
            if epub_to_audiobook.chapter_output_path(output_dir, c).exists()
        }

    def _refresh_status(self) -> None:
        done = self._done_chapter_numbers()
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            number = self.table.item(row, COL_CHAPTER).text()
            if number.isdigit():
                self.table.item(row, COL_STATUS).setText("done" if int(number) in done else "")
        self.table.blockSignals(False)

    def _refresh_preview(self, *_unused) -> None:
        searching = bool(self.search_hits)
        self.preview_box.setTitle("Search result" if searching else "Preview of selected chapters")
        if self.spine_docs is None:
            return  # keep the loading/error message
        if searching:
            self.preview.setHtml(self._search_hit_html())
            self.preview.scrollToAnchor("hit")
            return
        selected = self.selected_chapters()
        if not selected:
            self.preview.setPlainText("No chapters selected.")
            return
        limit = self.preview_spin.value()
        parts = []
        for chapter in selected:
            parts.append(
                f'<h3><a name="ch{chapter.number}"></a>{chapter.number} · {html.escape(chapter.title)} '
                f'<span style="color: gray; font-weight: normal">({chapter.word_count:,} words)</span></h3>'
            )
            for paragraph in chapter.paragraphs[:limit]:
                if len(paragraph) > PREVIEW_MAX_PARAGRAPH_CHARS:
                    paragraph = paragraph[:PREVIEW_MAX_PARAGRAPH_CHARS].rstrip() + "…"
                parts.append(f"<p>{html.escape(paragraph)}</p>")
        scrollbar = self.preview.verticalScrollBar()
        position = scrollbar.value()
        self.preview.setHtml("".join(parts))
        scrollbar.setValue(position)

    # ---- File pickers and drag-and-drop ----------------------------------

    def _browse_epub(self) -> None:
        start_dir = str(Path(self.epub_path).parent) if self.epub_path else str(CLI_SCRIPT.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Choose an EPUB", start_dir, "EPUB books (*.epub);;All files (*)")
        if path:
            self.set_epub(path)

    def _browse_output(self) -> None:
        current = self.output_dir()
        start_dir = str(current.parent) if current else str(CLI_SCRIPT.parent)
        path = QFileDialog.getExistingDirectory(self, "Choose the output directory", start_dir)
        if path:
            self.output_edit.setText(path)

    def _browse_voice(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a reference audio file", str(tts_synth.VOICES_DIR))
        if path:
            self._select_voice(path)

    def _select_voice(self, value: str) -> None:
        index = self.voice_combo.findData(value)
        if index < 0:
            if not Path(value).is_file():
                return  # a remembered voice that no longer exists
            self.voice_combo.addItem(value, value)
            index = self.voice_combo.count() - 1
        self.voice_combo.setCurrentIndex(index)

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].toLocalFile().lower().endswith(".epub"):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if self.process is None:
            self.set_epub(event.mimeData().urls()[0].toLocalFile())

    # ---- Running the CLI -------------------------------------------------

    def start(self) -> None:
        if self.process is not None or not self.selected_chapters():
            return
        state = self.form_state()
        done = set() if state.force else self._done_chapter_numbers()
        planned = [c for c in self.selected_chapters() if c.number not in done]
        self._planned = len(planned)
        self._started = 0
        self._out_buffer = ""
        # The bar covers the whole run, so count the chunks the CLI is about to make.
        total_chunks = sum(
            len(text_chunker.chunk_chapter(c.paragraphs, max_chars=state.max_chunk_chars)) for c in planned
        )
        self.progress.setRange(0, max(total_chunks, 1))
        self.progress.setValue(0)
        self.progress_label.setText("")
        self._run_started_at = time.monotonic()
        self._synth_started_at = None
        self._synth_count = 0
        self._eta_at = None
        self._update_eta_text()
        self._eta_timer.start()

        args = ["-u", str(CLI_SCRIPT), *build_cli_args(state)]
        self.log.appendPlainText("$ " + shlex.join([sys.executable, *args]))

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_process_error)
        self.process = process
        self._set_running(True)
        process.start(sys.executable, args)

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        self.log.appendPlainText("[gui] stopping…")
        process.terminate()
        QTimer.singleShot(
            STOP_KILL_TIMEOUT_MS,
            lambda: process.kill() if self.process is process else None,
        )

    def _set_running(self, running: bool) -> None:
        self.book_box.setEnabled(not running)
        self.options_box.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.start_button.setEnabled(not running and bool(self.selected_chapters()))

    def _on_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._out_buffer += data
        *lines, self._out_buffer = self._out_buffer.split("\n")
        for line in lines:
            # tqdm redraws its bar with \r; keep only the final state of each bar.
            self._handle_line(line.rstrip("\r").rsplit("\r", 1)[-1])

    def _handle_line(self, line: str) -> None:
        self.log.appendPlainText(line)
        match = _PROGRESS_RE.search(line)
        if match:
            chapter, chunk, total = (int(g) for g in match.groups()[:3])
            self.progress.setValue(min(self.progress.value() + 1, self.progress.maximum()))
            self.progress_label.setText(
                f"chapter {chapter} ({self._started} of {self._planned}) · chunk {chunk}/{total}"
            )
            if match.group(4) == "synthesized":
                # Cached chunks cost no time, so only synthesized ones set the pace.
                now = time.monotonic()
                self._synth_count += 1
                per_chunk = (now - self._synth_started_at) / self._synth_count
                self._eta_at = now + per_chunk * (self.progress.maximum() - self.progress.value())
            self._update_eta_text()
        elif line.startswith("[start]"):
            self._started += 1
            if self._synth_started_at is None:
                self._synth_started_at = time.monotonic()
        elif line.startswith("[done]"):
            self._refresh_status()

    def _update_eta_text(self) -> None:
        if self._eta_at is None:
            eta = "estimating…"
        else:
            eta = format_duration(max(0.0, self._eta_at - time.monotonic()))
        self.progress.setFormat(f"%p% · ETA {eta}")

    def _on_process_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(f"[gui] could not start {sys.executable}")
            self._eta_timer.stop()
            self.progress.setFormat("%p% · failed")
            self.process = None
            self._set_running(False)

    def _on_finished(self, exit_code: int, exit_status) -> None:
        if self._out_buffer:
            self._handle_line(self._out_buffer)
            self._out_buffer = ""
        self._eta_timer.stop()
        took = format_duration(time.monotonic() - self._run_started_at)
        if exit_status == QProcess.ExitStatus.CrashExit:
            self.log.appendPlainText(f"[gui] stopped after {took}")
            self.progress.setFormat("%p% · stopped")
        elif exit_code == 0:
            self.log.appendPlainText(f"[gui] finished in {took}")
            self.progress.setValue(self.progress.maximum())
            if self._planned:
                self.progress.setFormat(f"Completed: Took {took} to convert")
            else:
                self.progress.setFormat("Completed: nothing to convert, the selected chapters already exist")
        else:
            self.log.appendPlainText(f"[gui] failed with exit code {exit_code} after {took}")
            self.progress.setFormat("%p% · failed")
        self.process.deleteLater()
        self.process = None
        self._set_running(False)
        self._refresh_status()

    # ---- Settings and shutdown -------------------------------------------

    def _settings(self) -> QSettings:
        return QSettings("epub-audiobook-tts", "epub_gui")

    def _restore_settings(self) -> None:
        s = self._settings()
        self.output_edit.setText(s.value("output_dir", "", str))
        self._select_voice(s.value("voice", "", str))
        self.device_combo.setCurrentText(s.value("device", cli_default("device"), str))
        self.exaggeration_spin.setValue(s.value("exaggeration", cli_default("exaggeration"), float))
        self.cfg_spin.setValue(s.value("cfg_weight", cli_default("cfg_weight"), float))
        self.chunk_spin.setValue(s.value("max_chunk_chars", cli_default("max_chunk_chars"), int))
        self.threshold_spin.setValue(s.value("skip_word_threshold", cli_default("skip_word_threshold"), int))
        self.preview_spin.setValue(s.value("preview_paragraphs", DEFAULT_PREVIEW_PARAGRAPHS, int))
        self.set_epub(s.value("epub_path", "", str))

    def _save_settings(self) -> None:
        s = self._settings()
        s.setValue("epub_path", self.epub_path)
        s.setValue("output_dir", self.output_edit.text().strip())
        s.setValue("voice", self.voice_combo.currentData())
        s.setValue("device", self.device_combo.currentText())
        s.setValue("exaggeration", self.exaggeration_spin.value())
        s.setValue("cfg_weight", self.cfg_spin.value())
        s.setValue("max_chunk_chars", self.chunk_spin.value())
        s.setValue("skip_word_threshold", self.threshold_spin.value())
        s.setValue("preview_paragraphs", self.preview_spin.value())

    def closeEvent(self, event) -> None:
        if self.process is not None:
            answer = QMessageBox.question(
                self, "Conversion running",
                "A conversion is still running. Stop it and quit?\n"
                "Finished chunks stay cached, so it can be resumed later.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            process = self.process
            process.finished.disconnect(self._on_finished)
            process.terminate()
            if not process.waitForFinished(STOP_KILL_TIMEOUT_MS):
                process.kill()
                process.waitForFinished(STOP_KILL_TIMEOUT_MS)
        self._save_settings()
        event.accept()


def main(argv=None) -> None:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    window = MainWindow()
    if len(argv) > 1:
        window.set_epub(argv[1])
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
