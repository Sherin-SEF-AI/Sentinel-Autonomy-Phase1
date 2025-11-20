"""Driver score display widget."""

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QPalette, QColor

from src.core.data_structures import DriverScore


class DriverScoreWidget(QWidget):
    """
    Widget displaying real-time driver behavior score.
    Shows overall score and component breakdowns.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("📊 Driver Score")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Overall score (large display)
        score_frame = QFrame()
        score_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        score_layout = QVBoxLayout(score_frame)

        self.overall_score_label = QLabel("--")
        self.overall_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_font = QFont()
        score_font.setPointSize(36)
        score_font.setBold(True)
        self.overall_score_label.setFont(score_font)
        score_layout.addWidget(self.overall_score_label)

        overall_text = QLabel("Overall Score")
        overall_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(overall_text)

        layout.addWidget(score_frame)

        # Component scores
        components_frame = QFrame()
        components_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        components_layout = QVBoxLayout(components_frame)

        components_title = QLabel("Component Scores")
        components_title.setStyleSheet("font-weight: bold;")
        components_layout.addWidget(components_title)

        # Attention score
        self.attention_bar = self._create_score_bar("Attention", "#4ade80")
        components_layout.addLayout(self.attention_bar)

        # Smoothness score
        self.smoothness_bar = self._create_score_bar("Smoothness", "#60a5fa")
        components_layout.addLayout(self.smoothness_bar)

        # Safety score
        self.safety_bar = self._create_score_bar("Safety", "#fbbf24")
        components_layout.addLayout(self.safety_bar)

        # Hazard response score
        self.hazard_response_bar = self._create_score_bar("Hazard Response", "#a78bfa")
        components_layout.addLayout(self.hazard_response_bar)

        layout.addWidget(components_frame)

        # Recent events
        events_frame = QFrame()
        events_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        events_layout = QVBoxLayout(events_frame)

        events_title = QLabel("Recent Events")
        events_title.setStyleSheet("font-weight: bold;")
        events_layout.addWidget(events_title)

        self.events_label = QLabel("No recent events")
        self.events_label.setWordWrap(True)
        self.events_label.setStyleSheet("color: #888;")
        events_layout.addWidget(self.events_label)

        layout.addWidget(events_frame)

        layout.addStretch()

    def _create_score_bar(self, name: str, color: str):
        """Create a score bar with label and progress bar."""
        layout = QVBoxLayout()
        layout.setSpacing(2)

        # Label with value
        label_layout = QHBoxLayout()
        name_label = QLabel(name)
        label_layout.addWidget(name_label)
        label_layout.addStretch()

        value_label = QLabel("--")
        value_label.setStyleSheet("font-weight: bold;")
        label_layout.addWidget(value_label)
        layout.addLayout(label_layout)

        # Progress bar
        progress = QProgressBar()
        progress.setMinimum(0)
        progress.setMaximum(100)
        progress.setValue(0)
        progress.setTextVisible(False)
        progress.setMaximumHeight(15)
        progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #444;
                border-radius: 3px;
                background-color: #2e2e2e;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(progress)

        # Store references
        setattr(self, f"{name.lower().replace(' ', '_')}_label", value_label)
        setattr(self, f"{name.lower().replace(' ', '_')}_progress", progress)

        return layout

    @pyqtSlot(object, name="updateDriverScore")
    def update_driver_score(self, score: DriverScore):
        """Update driver score display."""
        if score is None:
            return

        # Update overall score
        overall = int(score.overall_score)
        self.overall_score_label.setText(f"{overall}")

        # Color based on score
        if overall >= 80:
            color = "#4ade80"  # Green
        elif overall >= 60:
            color = "#fbbf24"  # Yellow
        elif overall >= 40:
            color = "#fb923c"  # Orange
        else:
            color = "#f87171"  # Red

        self.overall_score_label.setStyleSheet(f"color: {color};")

        # Update component scores
        self._update_component("attention", score.attention_score)
        self._update_component("smoothness", score.smoothness_score)
        self._update_component("safety", score.safety_score)
        self._update_component("hazard_response", score.hazard_response_score)

        # Update recent events
        if score.recent_events:
            events_text = []
            for event in score.recent_events[-5:]:  # Show last 5
                event_type = event.get('type', 'unknown')
                severity = event.get('severity', 'low')

                # Format event name
                event_name = event_type.replace('_', ' ').title()

                # Emoji based on severity
                if severity == 'critical':
                    emoji = "🚨"
                elif severity == 'high':
                    emoji = "⚠️"
                elif severity == 'medium':
                    emoji = "⚡"
                else:
                    emoji = "ℹ️"

                events_text.append(f"{emoji} {event_name}")

            self.events_label.setText("\n".join(events_text))
            self.events_label.setStyleSheet("color: #fff;")
        else:
            self.events_label.setText("No recent events")
            self.events_label.setStyleSheet("color: #888;")

    def _update_component(self, component: str, value: float):
        """Update a component score bar."""
        label = getattr(self, f"{component}_label")
        progress = getattr(self, f"{component}_progress")

        label.setText(f"{int(value)}")
        progress.setValue(int(value))
