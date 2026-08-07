"""Interactive plotting tool.

Upload any number of diffraction patterns — in any of the formats the
converter understands (.xy, .dat, .txt, .csv, .xrdml, .ras, .rasx, .raw) —
overlay them in one interactive plot, and stack them vertically with a single
click. Files may be mixed: a .ras and a .csv can be plotted side by side.
"""

import io
import re
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from xrd_conversion import timestamp_suffix
from xrd_parsers import parse_xrdml, parse_ras, parse_rasx, parse_raw

# Colour-blind friendly qualitative palette, reused cyclically.
PALETTE = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b',
    '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#000000', '#aec7e8',
]

TEXT_EXTENSIONS = ('xy', 'dat', 'txt', 'csv', 'asc', 'prn')
SUPPORTED_EXTENSIONS = ["xy", "dat", "txt", "csv", "asc", "prn",
                        "xrdml", "xml", "ras", "rasx", "raw"]

NORMALIZATIONS = [
    "Raw data (no scaling)",
    "Normalize to max = 100",
    "Normalize to max = 1",
    "Min–max scaled (0–100)",
    "Unit area (∫y dx = 1)",
]


# ──────────────────────────────────────────────────────────────────────────
#  File reading
# ──────────────────────────────────────────────────────────────────────────
def _parse_text_columns(text):
    """Read numeric columns out of a free-form text/CSV table.

    Handles space, tab, comma and semicolon separators, comment lines, header
    rows and trailing text on a data line. Every row that starts with at least
    two numbers is kept; shorter rows are padded with NaN so that ragged files
    still produce a rectangular table.
    """
    rows = []
    width = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line[0] in "#!%*;'\"&/":
            continue
        values = []
        for part in re.split(r"[\s,;]+", line):
            try:
                values.append(float(part))
            except ValueError:
                break
        if len(values) < 2:
            continue
        rows.append(values)
        width = max(width, len(values))

    if not rows:
        return None

    padded = [row + [np.nan] * (width - len(row)) for row in rows]
    columns = [f"Column {i + 1}" for i in range(width)]
    return pd.DataFrame(padded, columns=columns)


@st.cache_data(show_spinner=False, max_entries=128)
def _load_table(file_name, file_bytes):
    """Return (DataFrame of numeric columns, error message).

    Cached on the file content, so moving a slider does not re-parse every
    uploaded file.
    """
    extension = file_name.lower().rsplit('.', 1)[-1]

    try:
        if extension in ('xrdml', 'xml'):
            _, data_df = parse_xrdml(file_bytes.decode("utf-8", errors='replace'))
        elif extension == 'ras':
            _, _, data_df = parse_ras(file_bytes.decode("utf-8", errors='replace'))
        elif extension == 'rasx':
            _, _, data_df = parse_rasx(BytesIO(file_bytes))
        elif extension == 'raw':
            _, data_df = parse_raw(file_bytes)
        elif extension in TEXT_EXTENSIONS:
            data_df = _parse_text_columns(file_bytes.decode("utf-8", errors='replace'))
        else:
            return None, f"Unsupported file type: .{extension}"
    except Exception as exc:                                   # noqa: BLE001
        return None, f"Could not read the file ({exc})."

    if data_df is None or len(data_df) == 0:
        return None, "No numeric data points were found in the file."

    return data_df.reset_index(drop=True), None


# ──────────────────────────────────────────────────────────────────────────
#  Processing
# ──────────────────────────────────────────────────────────────────────────
def _normalize(y_data, x_data, method):
    y_data = np.asarray(y_data, dtype=float)

    if method == "Normalize to max = 100":
        peak = np.nanmax(y_data)
        return y_data / peak * 100.0 if peak else y_data
    if method == "Normalize to max = 1":
        peak = np.nanmax(y_data)
        return y_data / peak if peak else y_data
    if method == "Min–max scaled (0–100)":
        low, high = np.nanmin(y_data), np.nanmax(y_data)
        return (y_data - low) / (high - low) * 100.0 if high > low else y_data - low
    if method == "Unit area (∫y dx = 1)":
        area = np.trapezoid(y_data, x_data) if hasattr(np, 'trapezoid') else np.trapz(y_data, x_data)
        return y_data / area if area else y_data
    return y_data


def _smooth(y_data, window):
    """Savitzky–Golay smoothing, with the window clamped to a valid size."""
    from scipy.signal import savgol_filter

    window = int(window)
    if window % 2 == 0:
        window += 1
    if window < 5 or window > len(y_data):
        window = min(len(y_data) - (1 - len(y_data) % 2), max(5, window))
    if window < 5 or window > len(y_data):
        return y_data
    return savgol_filter(y_data, window, 3)


def _number(value, default):
    """Read a number out of an editable table cell, tolerating a cleared cell."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(number) else number


def _default_y_label(normalization, stacked):
    if normalization.startswith("Normalize") or normalization.startswith("Min–max"):
        label = "Normalized intensity"
    elif normalization.startswith("Unit area"):
        label = "Intensity (unit area)"
    else:
        label = "Intensity (counts)"
    return f"{label} + offset" if stacked else label


# ──────────────────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────────────────
def run_plotting_section():
    # Same button colours as the File Format Converter page.
    st.markdown(
        """
        <style>
        button[data-testid^="stBaseButton-primary"] {
            background-color: #3b82f6; border-color: #3b82f6; color: #ffffff;
        }
        button[data-testid^="stBaseButton-primary"]:hover {
            background-color: #2563eb; border-color: #2563eb; color: #ffffff;
        }
        [data-testid="stDownloadButton"] button {
            background-color: #0e4d92; border-color: #0e4d92; color: #ffffff;
        }
        [data-testid="stDownloadButton"] button:hover {
            background-color: #0a3a6e; border-color: #0a3a6e; color: #ffffff;
        }
        .st-key-remove_all_files_plot button[data-testid^="stBaseButton-primary"] {
            background-color: #9ca3af; border-color: #9ca3af; color: #ffffff;
        }
        .st-key-remove_all_files_plot button[data-testid^="stBaseButton-primary"]:hover {
            background-color: #868e96; border-color: #868e96; color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "📈 Upload **one or more** diffraction patterns and compare them in one interactive plot. "
        "Supported: **.xy, .dat, .txt, .csv, .xrdml, .ras, .rasx, .raw** — and unlike the converter, "
        "you may **mix formats** in a single plot.\n\n"
        "📚 With two or more files the curves are **stacked automatically**; use the *Gap* slider to "
        "spread them out, or switch stacking off to overlay them."
    )

    if "plot_uploader_key" not in st.session_state:
        st.session_state.plot_uploader_key = 0

    def _clear_plot_files():
        st.session_state.plot_uploader_key += 1

    uploaded_files = st.file_uploader(
        "Upload Data File(s)",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key=f"plot_file_uploader_{st.session_state.plot_uploader_key}",
    )

    if not uploaded_files:
        st.markdown(
            "#### 📂 Upload your data files above to see them in an interactive plot.\n"
            "Two-column data separated by spaces, tabs, commas or semicolons is read "
            "automatically — header and comment lines are skipped for you."
        )
        return

    message_col, clear_col = st.columns([4, 1])
    with message_col:
        formats = sorted({f.name.lower().rsplit('.', 1)[-1] for f in uploaded_files})
        st.success(
            f"✅ Loaded **{len(uploaded_files)}** file(s) "
            f"(**{', '.join('.' + e for e in formats)}**)."
        )
    with clear_col:
        st.button("🗑️ Remove all files", key="remove_all_files_plot", type="primary",
                  width='stretch', on_click=_clear_plot_files)

    # A signature of the current file set: widget keys derived from it reset
    # the per-curve controls whenever the uploaded files change.
    signature = str(abs(hash(tuple(f.name for f in uploaded_files))))[:10]

    # ── Read every file ──────────────────────────────────────────────────
    # The parsers report their own progress with st.* messages, so the whole
    # reading step happens inside the (collapsed) log expander.
    tables, failed = [], []
    with st.expander("📄 File reading log", expanded=False):
        for uploaded_file in uploaded_files:
            data_df, error = _load_table(uploaded_file.name, uploaded_file.getvalue())
            if error:
                failed.append(uploaded_file.name)
                st.error(f"**{uploaded_file.name}**: {error}")
            else:
                st.write(f"**{uploaded_file.name}** — {len(data_df)} points, "
                         f"{len(data_df.columns)} numeric column(s).")
            tables.append(data_df)

    if all(table is None for table in tables):
        st.error("None of the uploaded files could be read.")
        return
    if failed:
        st.warning(f"⚠️ {len(failed)} file(s) could not be read and are not plotted: "
                   f"{', '.join(failed)} — see the *File reading log* above for details.")

    # ── Column choice for multi-column text files ────────────────────────
    column_choice = {}
    multi_column = [i for i, t in enumerate(tables)
                    if t is not None and len(t.columns) > 2]
    if multi_column:
        with st.expander(f"🔢 Choose columns ({len(multi_column)} file(s) have more than two columns)",
                         expanded=True):
            for i in multi_column:
                columns = list(tables[i].columns)
                name_col, x_col, y_col = st.columns([2, 1, 1])
                name_col.markdown(f"**{uploaded_files[i].name}**")
                x_choice = x_col.selectbox("X column", columns, index=0,
                                           key=f"plot_xcol_{i}_{signature}")
                y_choice = y_col.selectbox("Y column", columns, index=1,
                                           key=f"plot_ycol_{i}_{signature}")
                column_choice[i] = (x_choice, y_choice)

    def _columns_of(index):
        """The (x, y) column names used for the file at ``index``."""
        if index in column_choice:
            return column_choice[index]
        return tables[index].columns[0], tables[index].columns[1]

    # Data extent along x, needed for the range slider further down.
    x_low, x_high = np.inf, -np.inf
    for i, table in enumerate(tables):
        if table is None:
            continue
        x_values = pd.to_numeric(table[_columns_of(i)[0]], errors='coerce').values
        if np.isfinite(x_values).any():
            x_low = min(x_low, float(np.nanmin(x_values)))
            x_high = max(x_high, float(np.nanmax(x_values)))
    if not np.isfinite(x_low) or not np.isfinite(x_high):
        st.error("The uploaded files contain no usable X values.")
        return

    # ── Quick controls ───────────────────────────────────────────────────
    st.markdown("#### 🎛️ Plot controls")
    st.caption("⬅️ Curve labels, colours, axis labels, fonts, smoothing and the X-axis "
               "range are in the **sidebar**, under *Plotting settings*.")
    quick1, quick2, quick3 = st.columns([1.5, 2, 1.2])

    with quick1:
        normalization = st.selectbox(
            "Intensity scaling", NORMALIZATIONS,
            index=NORMALIZATIONS.index("Min–max scaled (0–100)"),
            key=f"plot_norm_{signature}",
            help="Normalizing puts patterns of very different count rates on a common scale.",
        )
    with quick2:
        stack = st.toggle("📚 Stack curves vertically", value=len(uploaded_files) > 1,
                          key=f"plot_stack_{signature}")
        gap = st.slider("Gap between curves (% of the tallest curve)", -300, 300, 135, 5,
                        key=f"plot_gap_{signature}", disabled=not stack,
                        help="0 puts every curve on the same baseline; negative values "
                             "stack downwards, so the first file ends up on top.")
    with quick3:
        log_y = st.toggle("Logarithmic Y-axis", key=f"plot_logy_{signature}")
        log_x = st.toggle("Logarithmic X-axis", key=f"plot_logx_{signature}")

    plot_placeholder = st.empty()

    # ── Per-curve appearance (sidebar, under the tool selector) ──────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Plotting settings")

    with st.sidebar.expander("🎨 Curves — labels, colours, order", expanded=False):
        reverse_order = st.checkbox(
            "First file on top of the stack", value=False,
            key=f"plot_reverse_{signature}", disabled=not stack,
            help="By default the first file sits at the bottom of the stack.")
        align_baselines = st.checkbox(
            "Align baselines before stacking", value=True,
            key=f"plot_align_{signature}", disabled=not stack,
            help="Shift each curve so that its minimum sits on its own baseline.")

        editor_df = pd.DataFrame({
            "Show": [True] * len(uploaded_files),
            "Label": [f.name for f in uploaded_files],
            "Scale": [1.0] * len(uploaded_files),
            "Extra offset": [0.0] * len(uploaded_files),
        })
        curve_settings = st.data_editor(
            editor_df,
            key=f"plot_curve_editor_{signature}",
            width='stretch',
            hide_index=True,
            column_config={
                "Show": st.column_config.CheckboxColumn("✓", width="small",
                                                        help="Show this curve in the plot."),
                "Label": st.column_config.TextColumn("Label", width="medium"),
                "Scale": st.column_config.NumberColumn(
                    "×", min_value=0.0, step=0.1, format="%.3f",
                    help="Multiply the intensities of this curve."),
                "Extra offset": st.column_config.NumberColumn(
                    "Offset", step=1.0, format="%.3f",
                    help="Added on top of the automatic stacking offset."),
            },
        )

        # A cleared "Label" cell falls back to the file name.
        labels = []
        for i, uploaded_file in enumerate(uploaded_files):
            label = curve_settings["Label"].iloc[i]
            labels.append(label.strip() if isinstance(label, str) and label.strip()
                          else uploaded_file.name)

        st.markdown("**Colours**")
        colors = {}
        color_columns = st.columns(2)
        for i in range(len(uploaded_files)):
            colors[i] = color_columns[i % 2].color_picker(
                labels[i][:18],
                value=PALETTE[i % len(PALETTE)],
                key=f"plot_color_{i}_{signature}",
            )

    # ── Axes, style and processing (sidebar) ─────────────────────────────
    with st.sidebar.expander("⚙️ Axes, style & smoothing", expanded=False):
        x_label = st.text_input("X-axis label", value="2θ (°)",
                                key=f"plot_xlab_{signature}")
        y_label = st.text_input("Y-axis label",
                                value=_default_y_label(normalization, stack),
                                key=f"plot_ylab_{signature}")
        plot_title = st.text_input("Plot title", value="",
                                   key=f"plot_title_{signature}")

        style1, style2 = st.columns(2)
        hover_mode = style1.selectbox("Hover", ["Nearest point", "Compare all curves"],
                                      key=f"plot_hover_{signature}")
        legend_position = style2.selectbox(
            "Legend", ["Top", "Right", "Bottom", "Inside (top right)", "Hidden"],
            key=f"plot_legend_{signature}")

        line_width = st.slider("Line width", 0.5, 6.0, 1.6, 0.1,
                               key=f"plot_lw_{signature}")
        marker1, marker2 = st.columns([1.2, 1])
        show_markers = marker1.checkbox("Markers", value=False,
                                        key=f"plot_markers_{signature}")
        marker_size = marker2.number_input("Size", 1, 12, 4,
                                           key=f"plot_ms_{signature}",
                                           disabled=not show_markers)

        plot_height = st.slider("Plot height (px)", 400, 1400, 700, 50,
                                key=f"plot_height_{signature}")
        font1, font2 = st.columns(2)
        axis_font = font1.number_input("Axis font", 10, 40, 22, 1,
                                       key=f"plot_axisfont_{signature}")
        tick_font = font2.number_input("Tick font", 8, 32, 16, 1,
                                       key=f"plot_tickfont_{signature}")

        smoothing = st.checkbox("Smooth curves", value=False,
                                key=f"plot_smooth_{signature}",
                                help="Savitzky–Golay filter (3rd order).")
        smooth_window = st.slider("Smoothing window (points)", 5, 201, 11, 2,
                                  key=f"plot_smoothwin_{signature}",
                                  disabled=not smoothing)

        limit_x = st.checkbox("Limit X-axis range", value=False,
                              key=f"plot_limitx_{signature}")
        if limit_x and x_high > x_low:
            x_range = st.slider("X-axis range", x_low, x_high, (x_low, x_high),
                                key=f"plot_xrange_{signature}",
                                help="Zooms the plot and trims the exported data.")
        else:
            x_range = None

    # ── Build the curves ─────────────────────────────────────────────────
    curves = []
    for i, uploaded_file in enumerate(uploaded_files):
        table = tables[i]
        if table is None or not bool(curve_settings["Show"].iloc[i]):
            continue

        x_name, y_name = _columns_of(i)
        x_data = pd.to_numeric(table[x_name], errors='coerce').values.astype(float)
        y_data = pd.to_numeric(table[y_name], errors='coerce').values.astype(float)

        valid = np.isfinite(x_data) & np.isfinite(y_data)
        x_data, y_data = x_data[valid], y_data[valid]
        if len(x_data) == 0:
            continue

        order = np.argsort(x_data)
        x_data, y_data = x_data[order], y_data[order]

        if smoothing:
            y_data = _smooth(y_data, smooth_window)

        y_data = _normalize(y_data, x_data, normalization)
        y_data = y_data * _number(curve_settings["Scale"].iloc[i], 1.0)

        curves.append({
            "index": i,
            "name": labels[i],
            "file_name": uploaded_file.name,
            "x": x_data,
            "y": y_data,
            "color": colors[i],
            "extra_offset": _number(curve_settings["Extra offset"].iloc[i], 0.0),
        })

    if not curves:
        st.warning("No curves to show — tick at least one file in the sidebar under "
                   "*Plotting settings → Curves*.")
        return

    # ── Stacking ─────────────────────────────────────────────────────────
    if stack and len(curves) > 1:
        if align_baselines:
            for curve in curves:
                curve["y"] = curve["y"] - np.nanmin(curve["y"])
        span = max((np.nanmax(c["y"]) - np.nanmin(c["y"])) for c in curves)
        step = (gap / 100.0) * (span if span > 0 else 1.0)
        count = len(curves)
        for position, curve in enumerate(curves):
            level = (count - 1 - position) if reverse_order else position
            curve["y"] = curve["y"] + level * step

    for curve in curves:
        curve["y"] = curve["y"] + curve["extra_offset"]

    # ── Figure ───────────────────────────────────────────────────────────
    mode = "lines+markers" if show_markers else "lines"
    figure = go.Figure()
    for curve in curves:
        figure.add_trace(go.Scatter(
            x=curve["x"], y=curve["y"], mode=mode, name=curve["name"],
            line=dict(width=line_width, color=curve["color"]),
            marker=dict(size=marker_size, color=curve["color"]),
            hovertemplate=f"<b>{curve['name']}</b><br>x = %{{x:.4f}}<br>y = %{{y:.4f}}<extra></extra>",
        ))

    legend_config = {"font": dict(size=max(12, tick_font))}
    if legend_position == "Top":
        legend_config.update(orientation="h", yanchor="bottom", y=1.02,
                             xanchor="center", x=0.5)
    elif legend_position == "Bottom":
        legend_config.update(orientation="h", yanchor="top", y=-0.18,
                             xanchor="center", x=0.5)
    elif legend_position == "Right":
        legend_config.update(orientation="v", yanchor="middle", y=0.5,
                             xanchor="left", x=1.02)
    elif legend_position == "Inside (top right)":
        legend_config.update(orientation="v", yanchor="top", y=0.98,
                             xanchor="right", x=0.98,
                             bgcolor="rgba(255,255,255,0.7)")

    figure.update_layout(
        height=plot_height,
        margin=dict(t=70 if plot_title or legend_position == "Top" else 40,
                    b=70, l=80, r=40),
        hovermode="x unified" if hover_mode == "Compare all curves" else "closest",
        showlegend=legend_position != "Hidden",
        legend=legend_config,
        title=dict(text=plot_title, font=dict(size=axis_font + 4)) if plot_title else None,
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=axis_font)),
            tickfont=dict(size=tick_font),
            type="log" if log_x else "linear",
            range=list(x_range) if x_range else None,
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=axis_font)),
            tickfont=dict(size=tick_font),
            type="log" if log_y else "linear",
        ),
        template="plotly_white",
    )

    with plot_placeholder.container():
        st.plotly_chart(
            figure,
            width='stretch',
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "toImageButtonOptions": {
                    "format": "png", "filename": f"xrd_plot_{timestamp_suffix()}", "scale": 3},
            },
        )
        st.caption(
            "🖱️ Drag to zoom, double-click to reset, click a legend entry to hide a curve. "
            "Use the 📷 camera icon in the plot toolbar to save a PNG image."
        )

    # ── Downloads ────────────────────────────────────────────────────────
    st.markdown("#### 💾 Download")
    download1, download2, download3 = st.columns([1.2, 1, 1])

    delimiters = {"Tab": "\t", "Space": " ", "Comma (,)": ",", "Semicolon (;)": ";"}
    delimiter_label = download1.selectbox("Delimiter for the exported data",
                                          list(delimiters.keys()),
                                          key=f"plot_delim_{signature}")
    delimiter = delimiters[delimiter_label]

    zip_buffer = BytesIO()
    used_names = set()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for curve in curves:
            x_data, y_data = curve["x"], curve["y"]
            if x_range:
                inside = (x_data >= x_range[0]) & (x_data <= x_range[1])
                x_data, y_data = x_data[inside], y_data[inside]
            text_buffer = io.StringIO()
            text_buffer.write(f"# {x_label}{delimiter}{y_label}\n")
            pd.DataFrame({"x": x_data, "y": y_data}).to_csv(
                text_buffer, sep=delimiter, header=False, index=False, float_format='%.6f')

            # Two uploads can share a base name (e.g. sample.ras and sample.xy),
            # so make the entry names unique inside the archive.
            entry_name = f"{curve['file_name'].rsplit('.', 1)[0]}_plotted.xy"
            if entry_name in used_names:
                entry_name = f"{curve['file_name'].rsplit('.', 1)[0]}_{curve['index'] + 1}_plotted.xy"
            used_names.add(entry_name)
            archive.writestr(entry_name, text_buffer.getvalue())

    with download2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.download_button(
            "📦 Download plotted data (.zip)",
            data=zip_buffer.getvalue(),
            file_name=f"plotted_data_{timestamp_suffix()}.zip",
            mime="application/zip",
            width='stretch',
        )

    # The self-contained HTML embeds plotly.js (~5 MB), so it is only built on
    # request and rebuilt whenever the plot changes.
    plot_state = str((
        signature, normalization, stack, gap, reverse_order, align_baselines,
        log_x, log_y, x_label, y_label, plot_title, legend_position, line_width,
        show_markers, marker_size, plot_height, axis_font, tick_font, smoothing,
        smooth_window, tuple(x_range) if x_range else None,
        tuple((c["name"], c["color"], len(c["x"]),
               round(float(np.nanmin(c["y"])), 6), round(float(np.nanmax(c["y"])), 6))
              for c in curves),
    ))

    with download3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.session_state.get("plot_html_state") == plot_state:
            st.download_button(
                "⬇️ Download interactive plot (.html)",
                data=st.session_state["plot_html_data"],
                file_name=f"xrd_plot_{timestamp_suffix()}.html",
                mime="text/html",
                width='stretch',
                help="A self-contained web page — the plot stays zoomable, no internet needed.",
            )
        elif st.button("🌐 Prepare interactive plot (.html)", width='stretch',
                       help="Builds a self-contained, zoomable web page of the plot above."):
            st.session_state["plot_html_data"] = figure.to_html(
                include_plotlyjs=True, full_html=True)
            st.session_state["plot_html_state"] = plot_state
            st.rerun()
