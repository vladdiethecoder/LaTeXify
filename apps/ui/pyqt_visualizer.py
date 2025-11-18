"""LaTeXify Pipeline Visualizer implemented with PyQt6 + pyqtgraph + QScintilla."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

try:  # pragma: no cover - optional at runtime
    from PyQt6.Qsci import QsciScintilla
except ImportError:  # pragma: no cover - fallback path
    QsciScintilla = None


COLORS: Dict[str, str] = {
    "background": "#1e1e1e",
    "text_primary": "#ffffff",
    "text_secondary": "#cccccc",
    "stage_complete": "#00ff00",
    "stage_active": "#ffff00",
    "stage_error": "#ff0000",
    "stage_pending": "#666666",
    "text_block": "#1e90ff",
    "math_block": "#32cd32",
    "table_block": "#ff8c00",
    "figure_block": "#9370db",
    "header_block": "#ffd700",
}


@dataclass
class PipelineState:
    current_stage: str
    stage_progress: float
    stage_start_time: datetime
    current_thought: str


@dataclass
class DocumentElement:
    element_id: str
    element_type: str  # text, math, table, figure, header
    bbox: Tuple[float, float, float, float]  # normalized (x, y, w, h)
    confidence: float
    latex_content: str
    line_count: int


@dataclass
class ProcessingMetrics:
    tokens_processed: int
    current_confidence: float
    processing_speed: float  # elements per second
    memory_usage: float  # MB


class StageIndicator(QFrame):
    """Displays a single stage with color-coded status."""

    def __init__(self, display_name: str):
        super().__init__()
        self.display_name = display_name
        self.status = "pending"
        self._message: Optional[str] = None
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        self.title_label = QLabel(display_name)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")

        self.status_label = QLabel("PENDING")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        self._apply_color(COLORS["stage_pending"])

    def _apply_color(self, color_hex: str) -> None:
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {color_hex};
                border-radius: 6px;
                border: 1px solid #111111;
            }}
            """
        )

    def set_status(self, status: str, timestamp: Optional[str] = None, message: Optional[str] = None) -> None:
        """Update the visual state of the stage indicator."""

        self.status = status
        self._message = message
        color_map = {
            "complete": COLORS["stage_complete"],
            "active": COLORS["stage_active"],
            "error": COLORS["stage_error"],
            "pending": COLORS["stage_pending"],
        }
        label = status.upper()
        if timestamp:
            label = f"{label} @ {timestamp}"
        if message:
            label = f"{label}\n{message}"
        self.status_label.setText(label)
        self._apply_color(color_map.get(status, COLORS["stage_pending"]))


class HolographicCanvas(pg.GraphicsLayoutWidget):
    """Canvas for rendering document elements as colored blocks."""

    def __init__(self):
        super().__init__()
        self.setBackground(COLORS["background"])
        self._view = self.addViewBox(lockAspect=False)
        self._view.setRange(xRange=(0, 1), yRange=(0, 1))
        self._view.invertY(True)  # origin at top-left
        self._elements: Dict[str, Tuple[QtWidgets.QGraphicsRectItem, pg.TextItem]] = {}
        self._init_grid_overlay()

    def _init_grid_overlay(self) -> None:
        """Add a faint grid to enhance the holographic feel."""

        pen = pg.mkPen(color=QColor("#333333"), width=1, style=Qt.PenStyle.DotLine)
        for fraction in [0.25, 0.5, 0.75]:
            vertical = pg.InfiniteLine(pos=fraction, angle=90, pen=pen)
            horizontal = pg.InfiniteLine(pos=fraction, angle=0, pen=pen)
            self._view.addItem(vertical)
            self._view.addItem(horizontal)

    def _block_color(self, element_type: str) -> str:
        mapping = {
            "text": COLORS["text_block"],
            "math": COLORS["math_block"],
            "table": COLORS["table_block"],
            "figure": COLORS["figure_block"],
            "header": COLORS["header_block"],
        }
        return mapping.get(element_type, COLORS["text_block"])

    def clear_elements(self) -> None:
        for rect, text in self._elements.values():
            self._view.removeItem(rect)
            self._view.removeItem(text)
        self._elements.clear()

    def add_element(self, element: DocumentElement) -> None:
        x, y, w, h = element.bbox
        rect_item = pg.QtGui.QGraphicsRectItem(x, y, w, h)
        color = QColor(self._block_color(element.element_type))
        color.setAlpha(120)
        rect_item.setBrush(pg.mkBrush(color))
        rect_item.setPen(pg.mkPen(color=color, width=2))
        self._view.addItem(rect_item)

        info = f"[{element.element_type.upper()}] {element.confidence*100:.0f}% • {element.line_count} lines"
        text_color = pg.mkColor(COLORS["text_primary"])
        text_item = pg.TextItem(info, color=text_color, anchor=(0, 1))
        text_item.setPos(x, max(0, y - 0.01))
        text_item.setTextWidth(w)
        self._view.addItem(text_item)

        self._elements[element.element_id] = (rect_item, text_item)


class LLMConsole(QWidget):
    """Monospace console for thoughts plus metrics dashboard."""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.thought_console = self._build_console()
        self.metrics_panel = self._build_metrics_panel()

        layout.addWidget(self.thought_console, stretch=3)
        layout.addWidget(self.metrics_panel, stretch=2)

    def _build_console(self) -> QWidget:
        font = QFont("Consolas", 10)
        if QsciScintilla is not None:
            widget = QsciScintilla()
            widget.setReadOnly(True)
            widget.setCaretLineVisible(False)
            widget.setMarginsVisible(False)
            widget.SendScintilla(widget.SCI_STYLESETBACK, widget.STYLE_DEFAULT, QColor(COLORS["background"]))
            widget.setPaper(QColor(COLORS["background"]))
            widget.setColor(QColor(COLORS["text_primary"]))
            widget.setFont(font)
        else:
            widget = QtWidgets.QPlainTextEdit()
            widget.setReadOnly(True)
            widget.setBackgroundVisible(False)
            widget.setFont(font)
            widget.setStyleSheet(
                f"background-color: {COLORS['background']}; color: {COLORS['text_primary']}; border: 1px solid #333333;"
            )
        widget.setMinimumHeight(150)
        return widget

    def _build_metrics_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet(f"background-color: #222222; border: 1px solid #444444;")
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        font = QFont("Consolas", 10)
        self.metric_labels: Dict[str, QLabel] = {}
        labels = ["Tokens Processed", "Confidence", "Speed (elem/s)", "Memory (MB)"]
        keys = ["tokens_processed", "current_confidence", "processing_speed", "memory_usage"]
        for row, (label_text, key) in enumerate(zip(labels, keys)):
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {COLORS['text_secondary']};")
            label.setFont(font)
            value_label = QLabel("--")
            value_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")
            value_label.setFont(font)
            layout.addWidget(label, row, 0, Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(value_label, row, 1, Qt.AlignmentFlag.AlignRight)
            self.metric_labels[key] = value_label
        return panel

    def append_thought(self, thought: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {thought}"
        if QsciScintilla is not None:
            current_text = self.thought_console.text()
            new_text = f"{current_text}\n{line}" if current_text else line
            self.thought_console.setText(new_text)
            self.thought_console.SendScintilla(self.thought_console.SCI_GOTOPOS, len(new_text))
        else:
            self.thought_console.appendPlainText(line)
            self.thought_console.verticalScrollBar().setValue(self.thought_console.verticalScrollBar().maximum())

    def update_metrics(self, metrics: ProcessingMetrics) -> None:
        self.metric_labels["tokens_processed"].setText(f"{metrics.tokens_processed:,}")
        self.metric_labels["current_confidence"].setText(f"{metrics.current_confidence:.2f}")
        self.metric_labels["processing_speed"].setText(f"{metrics.processing_speed:.2f}")
        self.metric_labels["memory_usage"].setText(f"{metrics.memory_usage:.1f}")

    def clear(self) -> None:
        if QsciScintilla is not None:
            self.thought_console.setText("")
        else:
            self.thought_console.clear()
        for label in self.metric_labels.values():
            label.setText("--")


class PipelineVisualizer(QMainWindow):
    """Main window for the LaTeXify Pipeline Visualizer."""

    STAGES: List[Tuple[str, str]] = [
        ("ingestion", "Ingestion"),
        ("dla", "DLA"),
        ("planning", "Planning"),
        ("rag", "RAG"),
        ("synthesis", "Synthesis"),
        ("assembly", "Assembly"),
        ("validation", "Validation"),
        ("visual_qa", "Visual QA"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LaTeXify Pipeline Visualizer v1.0")
        self.setFixedSize(1200, 800)
        self.setStyleSheet(f"background-color: {COLORS['background']}; color: {COLORS['text_primary']};")

        self.stage_indicators: Dict[str, StageIndicator] = {}
        self.current_stage: Optional[str] = None

        self._build_layout()

    def _build_layout(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        self.progress_section = self._build_progress_section()
        self.document_section = self._build_document_section()
        self.llm_section = self._build_llm_section()

        layout.addWidget(self.progress_section, stretch=2)
        layout.addWidget(self.document_section, stretch=5)
        layout.addWidget(self.llm_section, stretch=3)
        self.setCentralWidget(central)

    def _build_progress_section(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444444;")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(12, 8, 12, 8)
        vbox.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                background-color: #111111;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            """
        )
        vbox.addWidget(self.progress_bar)

        stages_row = QHBoxLayout()
        stages_row.setSpacing(8)
        for key, title in self.STAGES:
            indicator = StageIndicator(title)
            self.stage_indicators[key] = indicator
            stages_row.addWidget(indicator)
        vbox.addLayout(stages_row)

        self.timestamp_label = QLabel("Waiting for pipeline…")
        self.timestamp_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.timestamp_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        vbox.addWidget(self.timestamp_label)
        return frame

    def _build_document_section(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("background-color: #151515; border: 1px solid #444444;")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(12, 8, 12, 8)
        vbox.setSpacing(6)
        title = QLabel("Holographic Document Layout")
        title.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")
        vbox.addWidget(title)

        self.document_canvas = HolographicCanvas()
        vbox.addWidget(self.document_canvas, stretch=1)
        return frame

    def _build_llm_section(self) -> QWidget:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444444;")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(12, 8, 12, 8)
        vbox.setSpacing(6)

        title = QLabel("LLM Thinking & Metrics")
        title.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")
        vbox.addWidget(title)

        self.llm_console = LLMConsole()
        vbox.addWidget(self.llm_console, stretch=1)
        return frame

    def _stage_index(self, stage_name: str) -> int:
        for idx, (key, _) in enumerate(self.STAGES):
            if key == stage_name:
                return idx
        return -1

    def update_pipeline_stage(self, stage_name: str, progress: float) -> None:
        """Update current pipeline stage and progress percentage."""

        stage_key = stage_name.lower()
        if stage_key not in self.stage_indicators:
            return
        self.current_stage = stage_key
        total_stages = len(self.STAGES)
        idx = self._stage_index(stage_key)
        timestamp = datetime.now().strftime("%H:%M:%S")

        for i, (key, _) in enumerate(self.STAGES):
            indicator = self.stage_indicators[key]
            if key == stage_key:
                indicator.set_status("active", timestamp=timestamp)
            elif i < idx and indicator.status != "error":
                indicator.set_status("complete")
            elif indicator.status != "error":
                indicator.set_status("pending")

        overall_progress = ((idx) + max(0.0, min(progress, 1.0))) / total_stages
        self.progress_bar.setValue(int(overall_progress * 100))
        self.timestamp_label.setText(f"Stage '{stage_name.title()}' updated at {timestamp}")

    def add_document_element(self, element: DocumentElement) -> None:
        """Add new document element to holographic display."""

        self.document_canvas.add_element(element)

    def update_llm_thought(self, thought: str, metrics: ProcessingMetrics) -> None:
        """Add new LLM thought to thinking display."""

        self.llm_console.append_thought(thought)
        self.llm_console.update_metrics(metrics)

    def set_stage_error(self, stage_name: str, error_message: str) -> None:
        """Mark a stage as errored with message."""

        stage_key = stage_name.lower()
        indicator = self.stage_indicators.get(stage_key)
        if indicator:
            timestamp = datetime.now().strftime("%H:%M:%S")
            indicator.set_status("error", timestamp=timestamp, message=error_message)
            self.timestamp_label.setText(f"Stage '{stage_name}' error at {timestamp}: {error_message}")

    def clear_visualization(self) -> None:
        """Reset all displays for new document."""

        self.progress_bar.setValue(0)
        for indicator in self.stage_indicators.values():
            indicator.set_status("pending")
        self.document_canvas.clear_elements()
        self.llm_console.clear()
        self.timestamp_label.setText("Visualization reset.")


def _run_sample_demo(window: PipelineVisualizer) -> None:
    """Feed the UI with sample data to verify behavior."""

    test_elements = [
        DocumentElement("elem1", "header", (0.1, 0.1, 0.8, 0.05), 0.95, "\\section{Introduction}", 1),
        DocumentElement("elem2", "text", (0.1, 0.2, 0.8, 0.1), 0.87, "This is a sample paragraph...", 3),
        DocumentElement("elem3", "math", (0.1, 0.35, 0.3, 0.05), 0.92, "\\[ E = mc^2 \\]", 1),
    ]

    test_thoughts = [
        "Processing document layout analysis...",
        "Detected mathematical equation with 92% confidence",
        "Generating LaTeX representation for table structure",
    ]

    metrics_sequence = [
        ProcessingMetrics(150, 0.82, 1.5, 220.0),
        ProcessingMetrics(320, 0.88, 1.8, 235.0),
        ProcessingMetrics(540, 0.91, 2.0, 240.0),
    ]

    stage_schedule = [stage for stage, _ in PipelineVisualizer.STAGES]
    stage_state = {"index": 0, "progress": 0.0}

    stage_timer = QtCore.QTimer()

    def stage_tick():
        if stage_state["index"] >= len(stage_schedule):
            stage_timer.stop()
            return
        stage_name = stage_schedule[stage_state["index"]]
        stage_state["progress"] += 0.25
        if stage_state["progress"] >= 1.0:
            window.update_pipeline_stage(stage_name, 1.0)
            stage_state["index"] += 1
            stage_state["progress"] = 0.0
        else:
            window.update_pipeline_stage(stage_name, stage_state["progress"])

    stage_timer.timeout.connect(stage_tick)
    stage_timer.start(500)

    element_timer = QtCore.QTimer()
    element_iter = iter(test_elements)

    def add_next_element():
        try:
            element = next(element_iter)
        except StopIteration:
            element_timer.stop()
            return
        window.add_document_element(element)

    element_timer.timeout.connect(add_next_element)
    element_timer.start(1200)

    thoughts_iter = iter(zip(test_thoughts, metrics_sequence))

    def add_thought():
        try:
            thought, metrics = next(thoughts_iter)
        except StopIteration:
            thought_timer.stop()
            return
        window.update_llm_thought(thought, metrics)

    thought_timer = QtCore.QTimer()
    thought_timer.timeout.connect(add_thought)
    thought_timer.start(1500)

    # Retain timers on the window to prevent premature garbage collection.
    window._demo_timers = [stage_timer, element_timer, thought_timer]


def main() -> None:
    """Entrypoint for running the visualizer."""

    try:
        app = QApplication(sys.argv)
    except Exception as exc:  # pragma: no cover - missing Qt installation
        raise RuntimeError(
            "PyQt6 is required (version 6.4.0). Install via `pip install PyQt6==6.4.0` and ensure a display server is available."
        ) from exc
    pg.setConfigOption("background", COLORS["background"])
    pg.setConfigOption("foreground", COLORS["text_primary"])

    window = PipelineVisualizer()
    window.show()
    _run_sample_demo(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
