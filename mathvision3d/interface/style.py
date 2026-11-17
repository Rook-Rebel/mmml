"""Dark scientific theme."""

BG = "#0d1117"
PANEL = "#141a23"
CARD = "#1a2230"
BORDER = "#263244"
TEXT = "#e6edf3"
MUTED = "#8b98a9"
ACCENT = "#4fc3f7"
WARN = "#ffb74d"
OK = "#81c784"

STYLESHEET = f"""
* {{ font-family: "Segoe UI", "Sylfaen", sans-serif; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {PANEL}; border: none; }}
QGroupBox {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    margin-top: 14px; padding: 10px 8px 8px 8px; font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT}; }}
QPushButton {{
    background: #1f2a3a; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 10px;
}}
QPushButton:hover {{ background: #263449; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #2d3e57; }}
QPushButton:checked {{ background: #1d4a63; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #556070; border-color: #1c2533; }}
QPushButton#primary {{ background: #0e5a7a; border-color: {ACCENT}; font-weight: 600; }}
QPushButton#primary:hover {{ background: #11709a; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: #0f141c; border: 1px solid {BORDER}; border-radius: 5px; padding: 3px 6px;
    selection-background-color: #1d4a63;
}}
QComboBox QAbstractItemView {{ background: #0f141c; border: 1px solid {BORDER}; }}
QListWidget {{ background: #0f141c; border: 1px solid {BORDER}; border-radius: 6px; }}
QListWidget::item {{ padding: 4px; }}
QListWidget::item:selected {{ background: #1d4a63; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; border-radius: 7px; background: {ACCENT}; }}
QProgressBar {{ background: #0f141c; border: 1px solid {BORDER}; border-radius: 5px; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QPlainTextEdit {{ background: #0a0e14; border: 1px solid {BORDER}; border-radius: 6px;
    font-family: "Cascadia Mono", "Consolas", "Sylfaen"; font-size: 9pt; color: #b8c4d2; }}
QSplitter::handle {{ background: {BORDER}; }}
QMenuBar {{ background: {PANEL}; }}
QMenuBar::item:selected {{ background: #1d4a63; }}
QMenu {{ background: {CARD}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: #1d4a63; }}
QLabel#title {{ font-size: 12pt; font-weight: 700; color: {TEXT}; }}
QLabel#equation {{ font-family: "Cambria Math", "Cambria", "Segoe UI"; font-size: 13pt; color: #ffd54f; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#warn {{ color: {WARN}; }}
QLabel#chip {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 2px 10px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; }}
QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; }}
"""
