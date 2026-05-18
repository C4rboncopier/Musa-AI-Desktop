"""Theme definitions for the Musa AI application.

Provides four professionally designed themes with full color palettes
for all UI components. Themes are selected at runtime and applied
globally via Qt stylesheets and Leaflet map CSS variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ThemeColors:
    """Complete color palette for a single theme."""

    # --- Application chrome ---
    window_bg: str
    window_fg: str
    sidebar_bg: str
    sidebar_border: str
    menubar_bg: str
    menubar_fg: str
    menubar_hover_bg: str
    menu_selected_bg: str
    statusbar_bg: str
    statusbar_fg: str
    statusbar_border: str

    # --- Typography ---
    title_fg: str
    subtitle_fg: str
    body_fg: str
    muted_fg: str

    # --- Cards ---
    card_bg: str
    card_border: str
    card_title_fg: str

    # --- Primary action ---
    primary_bg: str
    primary_fg: str
    primary_border: str
    primary_hover_bg: str

    # --- Secondary action ---
    secondary_bg: str
    secondary_fg: str
    secondary_border: str
    secondary_hover_bg: str
    secondary_hover_border: str

    # --- Form controls ---
    checkbox_border: str
    checkbox_bg: str
    checkbox_checked_bg: str
    checkbox_checked_border: str
    slider_groove: str
    slider_fill: str
    slider_handle_bg: str
    slider_handle_border: str

    # --- Metadata ---
    meta_key_fg: str
    meta_value_fg: str

    # --- Map area ---
    map_bg: str

    # --- Accent for extent outline ---
    extent_color: str

    # --- Scrollbar ---
    scrollbar_bg: str
    scrollbar_handle: str
    scrollbar_handle_hover: str

    # --- Separator ---
    separator_color: str

    # --- Theme selector button ---
    theme_btn_bg: str
    theme_btn_fg: str
    theme_btn_border: str
    theme_btn_hover_bg: str

    # --- Tooltip / overlay label ---
    tooltip_bg: str
    tooltip_border: str
    tooltip_fg: str

    # --- Leaflet controls ---
    leaflet_ctrl_bg: str
    leaflet_ctrl_fg: str
    leaflet_attr_bg: str
    leaflet_attr_fg: str


# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------

OBSIDIAN = ThemeColors(
    window_bg="#0a0f1a",
    window_fg="#e2e8f0",
    sidebar_bg="#0e1525",
    sidebar_border="#1a2744",
    menubar_bg="#0c1220",
    menubar_fg="#c8d6e5",
    menubar_hover_bg="#15213a",
    menu_selected_bg="#2563eb",
    statusbar_bg="#070c16",
    statusbar_fg="#94a3b8",
    statusbar_border="#162033",

    title_fg="#f1f5f9",
    subtitle_fg="#64748b",
    body_fg="#94a3b8",
    muted_fg="#475569",

    card_bg="#111b2e",
    card_border="#1c2d4a",
    card_title_fg="#e2e8f0",

    primary_bg="#2563eb",
    primary_fg="#e0ecff",
    primary_border="#3b82f6",
    primary_hover_bg="#3b82f6",

    secondary_bg="#131e33",
    secondary_fg="#c8d6e5",
    secondary_border="#24375a",
    secondary_hover_bg="#182844",
    secondary_hover_border="#3b82f6",

    checkbox_border="#334155",
    checkbox_bg="#0a0f1a",
    checkbox_checked_bg="#2563eb",
    checkbox_checked_border="#3b82f6",
    slider_groove="#1e2d47",
    slider_fill="#3b82f6",
    slider_handle_bg="#e2e8f0",
    slider_handle_border="#2563eb",

    meta_key_fg="#60a5fa",
    meta_value_fg="#cbd5e1",

    map_bg="#0a0f1a",

    extent_color="#3b82f6",

    scrollbar_bg="#0e1525",
    scrollbar_handle="#1e2d47",
    scrollbar_handle_hover="#2a3f61",

    separator_color="#1a2744",

    theme_btn_bg="#131e33",
    theme_btn_fg="#94a3b8",
    theme_btn_border="#24375a",
    theme_btn_hover_bg="#182844",

    tooltip_bg="rgba(14, 21, 37, 0.92)",
    tooltip_border="rgba(59, 130, 246, 0.45)",
    tooltip_fg="#dbeafe",

    leaflet_ctrl_bg="#131e33",
    leaflet_ctrl_fg="#c8d6e5",
    leaflet_attr_bg="rgba(14, 21, 37, 0.85)",
    leaflet_attr_fg="#94a3b8",
)

ARCTIC = ThemeColors(
    window_bg="#f4f6f9",
    window_fg="#1e293b",
    sidebar_bg="#ffffff",
    sidebar_border="#e2e8f0",
    menubar_bg="#f8fafc",
    menubar_fg="#334155",
    menubar_hover_bg="#e2e8f0",
    menu_selected_bg="#2563eb",
    statusbar_bg="#f8fafc",
    statusbar_fg="#64748b",
    statusbar_border="#e2e8f0",

    title_fg="#0f172a",
    subtitle_fg="#64748b",
    body_fg="#475569",
    muted_fg="#94a3b8",

    card_bg="#f8fafc",
    card_border="#e2e8f0",
    card_title_fg="#1e293b",

    primary_bg="#2563eb",
    primary_fg="#ffffff",
    primary_border="#3b82f6",
    primary_hover_bg="#1d4ed8",

    secondary_bg="#f1f5f9",
    secondary_fg="#334155",
    secondary_border="#cbd5e1",
    secondary_hover_bg="#e2e8f0",
    secondary_hover_border="#2563eb",

    checkbox_border="#94a3b8",
    checkbox_bg="#ffffff",
    checkbox_checked_bg="#2563eb",
    checkbox_checked_border="#1d4ed8",
    slider_groove="#e2e8f0",
    slider_fill="#2563eb",
    slider_handle_bg="#ffffff",
    slider_handle_border="#2563eb",

    meta_key_fg="#2563eb",
    meta_value_fg="#334155",

    map_bg="#f4f6f9",

    extent_color="#2563eb",

    scrollbar_bg="#f1f5f9",
    scrollbar_handle="#cbd5e1",
    scrollbar_handle_hover="#94a3b8",

    separator_color="#e2e8f0",

    theme_btn_bg="#f1f5f9",
    theme_btn_fg="#475569",
    theme_btn_border="#cbd5e1",
    theme_btn_hover_bg="#e2e8f0",

    tooltip_bg="rgba(255, 255, 255, 0.95)",
    tooltip_border="rgba(37, 99, 235, 0.35)",
    tooltip_fg="#1e293b",

    leaflet_ctrl_bg="#ffffff",
    leaflet_ctrl_fg="#334155",
    leaflet_attr_bg="rgba(255, 255, 255, 0.88)",
    leaflet_attr_fg="#64748b",
)

EVERGREEN = ThemeColors(
    window_bg="#0c1410",
    window_fg="#d4e7d0",
    sidebar_bg="#0f1a14",
    sidebar_border="#1a3326",
    menubar_bg="#0d1711",
    menubar_fg="#a3c9a0",
    menubar_hover_bg="#142a1e",
    menu_selected_bg="#16a34a",
    statusbar_bg="#091210",
    statusbar_fg="#6b9c6b",
    statusbar_border="#14291e",

    title_fg="#e8f5e4",
    subtitle_fg="#5a8a5a",
    body_fg="#8ab888",
    muted_fg="#3d6a3d",

    card_bg="#111f17",
    card_border="#1c3528",
    card_title_fg="#d4e7d0",

    primary_bg="#16a34a",
    primary_fg="#ecfdf5",
    primary_border="#22c55e",
    primary_hover_bg="#22c55e",

    secondary_bg="#0f1e15",
    secondary_fg="#a3c9a0",
    secondary_border="#1c3528",
    secondary_hover_bg="#142a1e",
    secondary_hover_border="#22c55e",

    checkbox_border="#2d5a3a",
    checkbox_bg="#0c1410",
    checkbox_checked_bg="#16a34a",
    checkbox_checked_border="#22c55e",
    slider_groove="#1a3326",
    slider_fill="#22c55e",
    slider_handle_bg="#e8f5e4",
    slider_handle_border="#16a34a",

    meta_key_fg="#4ade80",
    meta_value_fg="#c3dac0",

    map_bg="#0c1410",

    extent_color="#22c55e",

    scrollbar_bg="#0f1a14",
    scrollbar_handle="#1a3326",
    scrollbar_handle_hover="#245838",

    separator_color="#1a3326",

    theme_btn_bg="#0f1e15",
    theme_btn_fg="#6b9c6b",
    theme_btn_border="#1c3528",
    theme_btn_hover_bg="#142a1e",

    tooltip_bg="rgba(15, 26, 20, 0.92)",
    tooltip_border="rgba(34, 197, 94, 0.45)",
    tooltip_fg="#ecfdf5",

    leaflet_ctrl_bg="#0f1e15",
    leaflet_ctrl_fg="#a3c9a0",
    leaflet_attr_bg="rgba(15, 26, 20, 0.85)",
    leaflet_attr_fg="#6b9c6b",
)

SLATE = ThemeColors(
    window_bg="#18181b",
    window_fg="#d4d4d8",
    sidebar_bg="#1f1f23",
    sidebar_border="#2e2e33",
    menubar_bg="#1b1b1f",
    menubar_fg="#a1a1aa",
    menubar_hover_bg="#27272a",
    menu_selected_bg="#a855f7",
    statusbar_bg="#131316",
    statusbar_fg="#71717a",
    statusbar_border="#27272a",

    title_fg="#fafafa",
    subtitle_fg="#71717a",
    body_fg="#a1a1aa",
    muted_fg="#52525b",

    card_bg="#212126",
    card_border="#2e2e33",
    card_title_fg="#e4e4e7",

    primary_bg="#a855f7",
    primary_fg="#faf5ff",
    primary_border="#c084fc",
    primary_hover_bg="#c084fc",

    secondary_bg="#212126",
    secondary_fg="#a1a1aa",
    secondary_border="#3f3f46",
    secondary_hover_bg="#27272a",
    secondary_hover_border="#a855f7",

    checkbox_border="#3f3f46",
    checkbox_bg="#18181b",
    checkbox_checked_bg="#a855f7",
    checkbox_checked_border="#c084fc",
    slider_groove="#27272a",
    slider_fill="#a855f7",
    slider_handle_bg="#fafafa",
    slider_handle_border="#a855f7",

    meta_key_fg="#c084fc",
    meta_value_fg="#d4d4d8",

    map_bg="#18181b",

    extent_color="#a855f7",

    scrollbar_bg="#1f1f23",
    scrollbar_handle="#2e2e33",
    scrollbar_handle_hover="#3f3f46",

    separator_color="#2e2e33",

    theme_btn_bg="#212126",
    theme_btn_fg="#71717a",
    theme_btn_border="#3f3f46",
    theme_btn_hover_bg="#27272a",

    tooltip_bg="rgba(31, 31, 35, 0.92)",
    tooltip_border="rgba(168, 85, 247, 0.45)",
    tooltip_fg="#faf5ff",

    leaflet_ctrl_bg="#212126",
    leaflet_ctrl_fg="#a1a1aa",
    leaflet_attr_bg="rgba(31, 31, 35, 0.85)",
    leaflet_attr_fg="#71717a",
)


THEMES: Dict[str, ThemeColors] = {
    "Obsidian": OBSIDIAN,
    "Arctic": ARCTIC,
    "Evergreen": EVERGREEN,
    "Slate": SLATE,
}

DEFAULT_THEME = "Obsidian"


def generate_stylesheet(theme: ThemeColors) -> str:
    """Build the complete Qt stylesheet string for the given theme."""
    t = theme
    return f"""
        /* ---- Application ---- */
        QMainWindow {{
            background: {t.window_bg};
            color: {t.window_fg};
        }}

        /* ---- Menu bar ---- */
        QMenuBar {{
            background: {t.menubar_bg};
            color: {t.menubar_fg};
            border-bottom: 1px solid {t.sidebar_border};
            padding: 4px 6px;
            font-size: 12px;
        }}
        QMenuBar::item {{
            padding: 5px 12px;
            border-radius: 4px;
        }}
        QMenuBar::item:selected {{
            background: {t.menubar_hover_bg};
            color: {t.title_fg};
        }}
        QMenu {{
            background: {t.card_bg};
            color: {t.window_fg};
            border: 1px solid {t.card_border};
            border-radius: 6px;
            padding: 4px 0;
        }}
        QMenu::item {{
            padding: 6px 28px 6px 16px;
        }}
        QMenu::item:selected {{
            background: {t.menu_selected_bg};
            color: #ffffff;
            border-radius: 4px;
        }}
        QMenu::separator {{
            height: 1px;
            background: {t.separator_color};
            margin: 4px 8px;
        }}

        /* ---- Sidebar ---- */
        #sidebar {{
            background: {t.sidebar_bg};
            border-right: 1px solid {t.sidebar_border};
        }}
        #appTitle {{
            color: {t.title_fg};
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.3px;
        }}
        #appSubtitle {{
            color: {t.subtitle_fg};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.8px;
        }}

        /* ---- Buttons ---- */
        QPushButton {{
            min-height: 38px;
            border-radius: 7px;
            font-weight: 600;
            font-size: 12px;
            text-align: left;
            padding: 0 14px;
        }}
        #primaryButton {{
            background: {t.primary_bg};
            color: {t.primary_fg};
            border: 1px solid {t.primary_border};
        }}
        #primaryButton:hover {{
            background: {t.primary_hover_bg};
        }}
        #primaryButton:pressed {{
            background: {t.primary_bg};
        }}
        #secondaryButton {{
            background: {t.secondary_bg};
            color: {t.secondary_fg};
            border: 1px solid {t.secondary_border};
        }}
        #secondaryButton:hover {{
            background: {t.secondary_hover_bg};
            border-color: {t.secondary_hover_border};
        }}

        /* ---- Cards ---- */
        #card {{
            background: {t.card_bg};
            border: 1px solid {t.card_border};
            border-radius: 8px;
        }}
        #cardTitle {{
            color: {t.card_title_fg};
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.4px;
        }}

        /* ---- Body text ---- */
        #bodyText {{
            color: {t.body_fg};
            font-size: 12px;
        }}

        /* ---- Metadata ---- */
        #metaKey {{
            color: {t.meta_key_fg};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.6px;
        }}
        #metaValue {{
            color: {t.meta_value_fg};
            font-size: 11px;
        }}
        #exportPathSummary {{
            background: {t.secondary_bg};
            border: 1px solid {t.secondary_border};
            border-radius: 6px;
        }}
        #exportPathName {{
            color: {t.title_fg};
            font-size: 12px;
            font-weight: 700;
        }}
        #exportPathLocation {{
            color: {t.body_fg};
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 10px;
        }}
        #compactButton {{
            background: transparent;
            border: 1px solid {t.secondary_border};
            border-radius: 5px;
            color: {t.secondary_fg};
            font-size: 11px;
            font-weight: 700;
            min-height: 24px;
            padding: 0 9px;
            text-align: center;
        }}
        #compactButton:hover {{
            background: {t.secondary_hover_bg};
            border-color: {t.secondary_hover_border};
        }}

        /* ---- Checkbox ---- */
        QCheckBox {{
            color: {t.meta_value_fg};
            spacing: 10px;
            font-size: 12px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {t.checkbox_border};
            background: {t.checkbox_bg};
        }}
        QCheckBox::indicator:checked {{
            background: {t.checkbox_checked_bg};
            border-color: {t.checkbox_checked_border};
        }}

        /* ---- Slider ---- */
        QSlider::groove:horizontal {{
            height: 6px;
            background: {t.slider_groove};
            border-radius: 3px;
        }}
        QSlider::sub-page:horizontal {{
            background: {t.slider_fill};
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {t.slider_handle_bg};
            border: 2px solid {t.slider_handle_border};
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}

        /* ---- Map ---- */
        #mapView {{
            background: {t.map_bg};
        }}

        /* ---- Status bar ---- */
        #statusBar {{
            background: {t.statusbar_bg};
            color: {t.statusbar_fg};
            border-top: 1px solid {t.statusbar_border};
            font-size: 11px;
        }}
        QStatusBar QLabel {{
            color: {t.statusbar_fg};
            padding: 0 6px;
            font-size: 11px;
        }}

        /* ---- Scrollbar ---- */
        QScrollArea {{
            border: none;
            background: transparent;
        }}
        QScrollBar:vertical {{
            background: {t.scrollbar_bg};
            width: 8px;
            border: none;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {t.scrollbar_handle};
            border-radius: 4px;
            min-height: 28px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {t.scrollbar_handle_hover};
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{
            background: transparent;
        }}

        /* ---- Separator ---- */
        #separator {{
            background: {t.separator_color};
            max-height: 1px;
            min-height: 1px;
        }}

        /* ---- Theme selector ---- */
        #themeButton {{
            background: {t.theme_btn_bg};
            color: {t.theme_btn_fg};
            border: 1px solid {t.theme_btn_border};
            min-height: 30px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            padding: 0 10px;
            text-align: center;
        }}
        #themeButton:hover {{
            background: {t.theme_btn_hover_bg};
            color: {t.title_fg};
        }}
        #themeButton:checked {{
            background: {t.primary_bg};
            color: {t.primary_fg};
            border-color: {t.primary_border};
        }}

        /* ---- Collapse toggle ---- */
        #collapseButton {{
            background: transparent;
            border: 1px solid {t.sidebar_border};
            border-radius: 6px;
            color: {t.subtitle_fg};
            min-height: 28px;
            max-height: 28px;
            min-width: 28px;
            max-width: 28px;
            font-size: 14px;
            font-weight: 700;
            padding: 0;
        }}
        #collapseButton:hover {{
            background: {t.card_bg};
            color: {t.title_fg};
        }}

        /* ---- Opacity value label ---- */
        #opacityValue {{
            color: {t.meta_key_fg};
            font-size: 12px;
            font-weight: 700;
        }}

        /* ---- Section header ---- */
        #sectionHeader {{
            color: {t.subtitle_fg};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.2px;
        }}

        /* ---- Info label ---- */
        #infoLabel {{
            color: {t.muted_fg};
            font-size: 11px;
        }}

        /* ---- Combobox ---- */
        QComboBox {{
            background: {t.secondary_bg};
            color: {t.secondary_fg};
            border: 1px solid {t.secondary_border};
            border-radius: 6px;
            padding: 4px 10px;
            min-height: 30px;
            font-size: 12px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 22px;
        }}
        QComboBox QAbstractItemView {{
            background: {t.card_bg};
            color: {t.window_fg};
            border: 1px solid {t.card_border};
            selection-background-color: {t.menu_selected_bg};
            selection-color: #ffffff;
        }}

        /* ---- Professional app shell ---- */
        #navRail {{
            background: {t.sidebar_bg};
            border-right: 1px solid {t.sidebar_border};
        }}
        #dashboardContent {{
            background: {t.window_bg};
        }}
        #pageTitle {{
            color: {t.title_fg};
            font-size: 26px;
            font-weight: 700;
        }}
        #pageSubtitle {{
            color: {t.body_fg};
            font-size: 12px;
        }}
        #panelTitle {{
            color: {t.title_fg};
            font-size: 15px;
            font-weight: 700;
        }}
        #workspaceTopbar {{
            background: {t.menubar_bg};
            border-bottom: 1px solid {t.sidebar_border};
        }}
        #workspaceTitle {{
            color: {t.title_fg};
            font-size: 16px;
            font-weight: 700;
        }}
        #workspaceSidePanel,
        #workspaceInspector {{
            background: {t.sidebar_bg};
            border-right: 1px solid {t.sidebar_border};
        }}
        #workspaceInspector {{
            border-right: none;
            border-left: 1px solid {t.sidebar_border};
        }}
        #mapShell {{
            background: {t.map_bg};
        }}
        #mapToolbar {{
            background: {t.card_bg};
            border-bottom: 1px solid {t.card_border};
        }}
        #mapToolbarLabel {{
            color: {t.meta_value_fg};
            font-size: 11px;
            font-family: "Cascadia Code", "Consolas", monospace;
        }}
        #mapToolButton {{
            background: {t.secondary_bg};
            color: {t.secondary_fg};
            border: 1px solid {t.secondary_border};
            border-radius: 5px;
            min-width: 34px;
            min-height: 28px;
            font-size: 12px;
            font-weight: 700;
        }}
        #mapToolButton:hover {{
            background: {t.secondary_hover_bg};
            border-color: {t.secondary_hover_border};
        }}
        #navButton {{
            background: transparent;
            color: {t.secondary_fg};
            border: 1px solid transparent;
            border-radius: 6px;
            min-height: 36px;
            text-align: left;
            padding: 0 12px;
            font-weight: 600;
        }}
        #navButton:hover {{
            background: {t.secondary_bg};
            border-color: {t.secondary_border};
            color: {t.title_fg};
        }}
        #searchBox,
        QLineEdit,
        QTextEdit {{
            background: {t.secondary_bg};
            color: {t.window_fg};
            border: 1px solid {t.secondary_border};
            border-radius: 6px;
            padding: 7px 10px;
            selection-background-color: {t.primary_bg};
        }}
        #searchBox {{
            min-width: 280px;
        }}
        QLineEdit:focus,
        QTextEdit:focus {{
            border-color: {t.primary_border};
        }}
        QTextEdit {{
            font-size: 12px;
        }}
        #dialogTitle {{
            color: {t.title_fg};
            font-size: 18px;
            font-weight: 700;
        }}

        /* ---- Dashboard project list ---- */
        #projectListHeader {{
            background: {t.menubar_bg};
            border: 1px solid {t.card_border};
            border-radius: 7px;
        }}
        #projectListHeaderText {{
            color: {t.subtitle_fg};
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.8px;
        }}
        #projectListRow {{
            background: {t.card_bg};
            border: 1px solid {t.card_border};
            border-radius: 8px;
        }}
        #projectListRow:hover {{
            border-color: {t.primary_border};
            background: {t.secondary_hover_bg};
        }}
        #projectListTitle {{
            color: {t.title_fg};
            font-size: 13px;
            font-weight: 700;
        }}
        #projectListDescription {{
            color: {t.body_fg};
            font-size: 11px;
        }}
        #projectListMetric {{
            color: {t.meta_value_fg};
            background: {t.secondary_bg};
            border: 1px solid {t.secondary_border};
            border-radius: 5px;
            padding: 5px 6px;
            font-size: 11px;
            font-weight: 700;
        }}
        #projectListMeta {{
            color: {t.body_fg};
            font-size: 11px;
            font-family: "Cascadia Code", "Consolas", monospace;
        }}
        #tableActionButton,
        #tableDangerButton {{
            min-height: 28px;
            max-height: 28px;
            border-radius: 5px;
            padding: 0 10px;
            font-size: 11px;
            font-weight: 700;
            text-align: center;
        }}
        #tableActionButton {{
            background: {t.secondary_bg};
            color: {t.secondary_fg};
            border: 1px solid {t.secondary_border};
        }}
        #tableActionButton:hover {{
            background: {t.secondary_hover_bg};
            border-color: {t.secondary_hover_border};
            color: {t.title_fg};
        }}
        #tableDangerButton {{
            background: transparent;
            color: #fca5a5;
            border: 1px solid rgba(248, 113, 113, 0.35);
        }}
        #tableDangerButton:hover {{
            background: rgba(153, 27, 27, 0.28);
            border-color: #ef4444;
            color: #fee2e2;
        }}

        /* ---- Legacy dashboard cards ---- */
        #projectCard {{
            background: {t.card_bg};
            border: 1px solid {t.card_border};
            border-radius: 8px;
        }}
        #projectCard:hover {{
            border-color: {t.primary_border};
            background: {t.secondary_hover_bg};
        }}
        #projectCardTitle {{
            color: {t.title_fg};
            font-size: 14px;
            font-weight: 700;
        }}
        #projectCardBody {{
            color: {t.body_fg};
            font-size: 12px;
        }}
        #projectMiniStat {{
            color: {t.meta_value_fg};
            background: {t.secondary_bg};
            border: 1px solid {t.secondary_border};
            border-radius: 4px;
            padding: 3px 7px;
            font-size: 10px;
            font-weight: 600;
        }}
        #metricCard {{
            background: {t.card_bg};
            border: 1px solid {t.card_border};
            border-radius: 8px;
        }}
        #metricValue {{
            color: {t.title_fg};
            font-size: 24px;
            font-weight: 700;
        }}

        /* ---- Asset rows and states ---- */
        #assetRow {{
            background: {t.secondary_bg};
            border: 1px solid {t.secondary_border};
            border-radius: 7px;
        }}
        #assetTitle {{
            color: {t.title_fg};
            font-size: 12px;
            font-weight: 700;
        }}
        #assetPath {{
            color: {t.body_fg};
            font-size: 10px;
            font-family: "Cascadia Code", "Consolas", monospace;
        }}
        #outputRow {{
            background: {t.secondary_bg};
            border: 1px solid {t.secondary_border};
            border-radius: 7px;
        }}
        #outputTitle {{
            color: {t.title_fg};
            font-size: 12px;
            font-weight: 700;
        }}
        #outputPath {{
            color: {t.body_fg};
            font-size: 10px;
            font-family: "Cascadia Code", "Consolas", monospace;
        }}
        #outputMeta {{
            color: {t.meta_value_fg};
            font-size: 10px;
        }}
        #miniActionButton,
        #miniDangerButton {{
            min-height: 24px;
            max-height: 24px;
            border-radius: 5px;
            padding: 0 8px;
            font-size: 10px;
            font-weight: 700;
            text-align: center;
        }}
        #miniActionButton {{
            background: {t.card_bg};
            color: {t.secondary_fg};
            border: 1px solid {t.secondary_border};
        }}
        #miniActionButton:hover {{
            background: {t.secondary_hover_bg};
            border-color: {t.secondary_hover_border};
            color: {t.title_fg};
        }}
        #miniDangerButton {{
            background: transparent;
            color: #fca5a5;
            border: 1px solid rgba(248, 113, 113, 0.35);
        }}
        #miniDangerButton:hover {{
            background: rgba(153, 27, 27, 0.28);
            border-color: #ef4444;
            color: #fee2e2;
        }}
        #statusPill {{
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 9px;
            font-weight: 800;
            text-transform: uppercase;
        }}
        #statusPill[tone="ok"] {{
            color: #dcfce7;
            background: #166534;
        }}
        #statusPill[tone="danger"] {{
            color: #fee2e2;
            background: #991b1b;
        }}
        #statusPill[tone="neutral"] {{
            color: {t.meta_value_fg};
            background: {t.secondary_bg};
        }}
        #emptyTitle {{
            color: {t.title_fg};
            font-size: 16px;
            font-weight: 700;
        }}

        /* ---- Tabs and console ---- */
        QTabWidget::pane {{
            border: 1px solid {t.card_border};
            border-radius: 8px;
            background: {t.sidebar_bg};
        }}
        QTabBar::tab {{
            background: transparent;
            color: {t.body_fg};
            padding: 8px 10px;
            border-bottom: 2px solid transparent;
            font-size: 11px;
            font-weight: 700;
        }}
        QTabBar::tab:selected {{
            color: {t.title_fg};
            border-bottom-color: {t.primary_border};
        }}
        #logConsole {{
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 11px;
            line-height: 1.35;
        }}
    """
