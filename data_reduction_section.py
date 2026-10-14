# data_reduction_section.py

import streamlit as st
from ui_style import apply_button_style
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from io import StringIO, BytesIO
import zipfile

from download_log import log_download
from xrd_conversion import AXIS_UPLOAD_EXTENSIONS, load_xy_file, timestamp_suffix


REDUCE_METHODS = [
    "Keep every N-th point",
    "Average every N points (binning)",
    "Resample to new step size",
]


def reduce_points(x, y, method, params):
    """Thin an XRD pattern to fewer data points.

    Returns new ``(x, y)`` arrays. Supported methods match ``REDUCE_METHODS``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if method == "Keep every N-th point":
        n = max(1, int(params.get("n", 2)))
        return x[::n], y[::n]

    elif method == "Average every N points (binning)":
        n = max(1, int(params.get("n", 2)))
        starts = np.arange(0, len(x), n)
        counts = np.diff(np.append(starts, len(x)))
        return (np.add.reduceat(x, starts) / counts,
                np.add.reduceat(y, starts) / counts)

    elif method == "Resample to new step size":
        step = float(params.get("step", 0.0))
        if step <= 0 or len(x) < 2:
            return x, y
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        x_new = np.arange(xs[0], xs[-1] + step * 1e-6, step)
        return x_new, np.interp(x_new, xs, ys)

    return x, y


def reduced_file_text(x, y, include_header, source_name, method, n_before):
    """Two-column, tab-separated text of a reduced pattern."""
    output = StringIO()
    if include_header:
        output.write(f"# Reduced from {source_name}: {method}, "
                     f"{n_before} -> {len(x)} points\n")
    pd.DataFrame({'X': x, 'Y': y}).to_csv(
        output, sep='\t', header=False, index=False, float_format='%.8g')
    return output.getvalue()


def format_size(n_bytes):
    """Human-readable file size."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} kB"
    return f"{n_bytes / 1024 ** 2:.2f} MB"


def size_change(before, after):
    """'before → after (−x%)' text, with the sign of the change."""
    pct = 100 * (after - before) / before
    sign = "−" if pct < 0 else "+"
    return (f"**{format_size(before)}** → **{format_size(after)}** "
            f"({sign}{abs(pct):.0f}%)")


def run_data_reduction_section():
    st.markdown("### ✂️ Data Point Reduction")
    st.info(
        "✂️ Reduce the number of data points in your XRD patterns, e.g. to "
        "shrink the file size: **keep every N-th point**, **average every N "
        "points** (binning, also reduces noise) or **resample to a new step "
        "size**. Upload single or multiple files - batch mode activates "
        "automatically."
    )

    # The uploader key holds a counter. Incrementing it (in the clear button
    # callback below) forces Streamlit to mount a fresh, empty file_uploader.
    if "reduce_uploader_key" not in st.session_state:
        st.session_state.reduce_uploader_key = 0

    def _clear_reduce_uploaded_files():
        st.session_state.reduce_uploader_key += 1

    uploaded_files = st.file_uploader(
        "Upload XRD Data File(s) (.xy, .txt, .dat, .csv, .xrdml, .ras, .rasx, .raw)",
        type=AXIS_UPLOAD_EXTENSIONS,
        accept_multiple_files=True,
        key=f"reduce_file_uploader_{st.session_state.reduce_uploader_key}"
    )

    if not uploaded_files:
        st.info("👆 Upload your data file(s) to begin")
        return

    is_batch = len(uploaded_files) > 1

    # Which file is shown in the preview. In batch mode a slider below the
    # plot lets the user switch files; its value is stored in session_state
    # so it can be read here (before the widget itself is rendered).
    preview_idx = 0
    if is_batch:
        names = [f.name for f in uploaded_files]
        stored_name = st.session_state.get("reduce_preview_file_name")
        if stored_name in names:
            preview_idx = names.index(stored_name)
        elif stored_name is not None:
            del st.session_state["reduce_preview_file_name"]

    first_file = uploaded_files[preview_idx]
    data_df = load_xy_file(first_file)

    if data_df is None:
        st.error("Failed to parse the uploaded file.")
        return

    col1, col2 = st.columns([1, 1.5])

    with col1:
        hdr_col, btn_col = st.columns([1.5, 1])
        with hdr_col:
            st.markdown("#### ⚙️ Reduction Settings")
        with btn_col:
            apply_button_style(["remove_all_files_reduce"])
            st.button("🗑️ Remove all files",
                      key="remove_all_files_reduce",
                      on_click=_clear_reduce_uploaded_files,
                      type="primary",
                      use_container_width=True)

        if is_batch:
            st.success(f"✅ **Batch mode active:** {len(uploaded_files)} files uploaded")
            st.caption("Settings will apply to all files")

        reduce_method = st.radio(
            "Reduction Method",
            REDUCE_METHODS,
            index=0,
            key="reduce_method",
        )

        reduce_params = {}
        if reduce_method in ("Keep every N-th point",
                             "Average every N points (binning)"):
            reduce_params["n"] = st.number_input(
                "N", min_value=2, max_value=1000, value=2, step=1,
                key="reduce_n",
                help="Keep every N-th point, or average each group of N "
                     "neighbouring points into one (binning also "
                     "reduces noise).")
        else:
            orig_step = float(np.median(np.abs(np.diff(data_df['X'].values)))) \
                if len(data_df) > 1 else 0.01
            st.caption(f"Current step size: **{orig_step:.6g}**"
                       + (" (preview file)" if is_batch else ""))
            reduce_params["step"] = st.number_input(
                "New step size (in units of the uploaded x-axis)",
                min_value=1e-6, value=round(2 * orig_step, 6) or 0.02,
                step=orig_step or 0.01, format="%.6f",
                key="reduce_step",
                help="Data are linearly interpolated onto an evenly "
                     "spaced grid with this step.")

        st.markdown("##### 📊 Preview")
        stack_col, points_col = st.columns(2)
        with stack_col:
            reduce_stack = st.checkbox(
                "Stack original and reduced data",
                value=True, key="reduce_stack")
        with points_col:
            reduce_show_points = st.checkbox(
                "Show data points",
                value=True, key="reduce_show_points")
        reduce_offset_pct = 0
        if reduce_stack:
            reduce_offset_pct = st.slider(
                "Vertical offset (% of max. intensity)",
                0, 200, 30, 5, key="reduce_offset_pct")

        reduced_x, reduced_y = reduce_points(
            data_df['X'].values, data_df['Y'].values,
            reduce_method, reduce_params)
        n_before, n_after = len(data_df), len(reduced_x)
        st.success(
            f"✅ {n_before} → {n_after} points "
            f"({100 * (1 - n_after / n_before):.0f}% fewer)"
            + (" for the preview file" if is_batch else ""))

        st.markdown("---")

        st.markdown("#### 💾 Download Options")

        include_header = st.checkbox("Include header in output file", value=True,
                                     key="reduce_include_header")

        # File size of the previewed file before and after the reduction.
        size_before = len(first_file.getvalue())
        size_after = len(reduced_file_text(
            reduced_x, reduced_y, include_header,
            first_file.name, reduce_method, n_before).encode("utf-8"))
        size_msg = f"💾 File size: {size_change(size_before, size_after)}"
        if first_file.name.lower().rsplit('.', 1)[-1] in ('xrdml', 'xml', 'ras', 'rasx', 'raw'):
            # A different input format: also give the size of the unreduced
            # data as .xy, which is the fair comparison for the reduction.
            size_xy = len(reduced_file_text(
                data_df['X'].values, data_df['Y'].values, include_header,
                first_file.name, reduce_method, n_before).encode("utf-8"))
            size_msg += ("  \nSame data as .xy without reduction: "
                         + size_change(size_xy, size_after))
        st.info(size_msg + (f"  \n_Shown for the preview file: {first_file.name}_"
                            if is_batch else ""))

        if is_batch:
            if st.button("📦 Prepare All Reduced Files for Download (.zip)",
                         type="primary",
                         width='stretch'):

                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for uploaded_file in uploaded_files:
                        df = load_xy_file(uploaded_file)
                        if df is None:
                            continue
                        x_data, y_data = reduce_points(
                            df['X'].values, df['Y'].values,
                            reduce_method, reduce_params)
                        new_filename = uploaded_file.name.rsplit('.', 1)[0] + '_reduced.xy'
                        zf.writestr(new_filename, reduced_file_text(
                            x_data, y_data, include_header,
                            uploaded_file.name, reduce_method, len(df)))

                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"reduced_xrd_files_{timestamp_suffix()}.zip",
                    mime="application/zip",
                    type="primary",
                    width='stretch',
                    on_click=log_download,
                    args=("Data Reduction",),
                )
        else:
            default_name = first_file.name.rsplit('.', 1)[0] + '_reduced.xy'
            download_filename = st.text_input("Output filename:", default_name,
                                              key="reduce_output_name")
            st.download_button(
                label="⬇️ Download Reduced File",
                data=reduced_file_text(reduced_x, reduced_y, include_header,
                                       first_file.name, reduce_method, n_before),
                file_name=download_filename,
                mime="text/plain",
                type="primary",
                width='stretch',
                on_click=log_download,
                args=("Data Reduction",),
            )

    with col2:
        st.markdown("#### 📊 Data Preview & Comparison")

        mode = 'lines+markers' if reduce_show_points else 'lines'
        reduce_offset = 0.0
        if reduce_stack:
            reduce_offset = reduce_offset_pct / 100 * float(np.nanmax(data_df['Y']))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_df['X'],
            y=data_df['Y'],
            mode=mode,
            name=f'Original ({n_before} pts)',
            line=dict(color='#0984e3', width=1),
            marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=reduced_x,
            y=reduced_y + reduce_offset,
            mode=mode,
            name=f'Reduced ({n_after} pts)'
                 + (f', offset +{reduce_offset:.4g}' if reduce_offset else ''),
            line=dict(color='#e17055', width=1.5),
            marker=dict(size=7, symbol='circle-open', line=dict(width=1.5))
        ))

        fig.update_layout(
            title=f"Preview: {first_file.name}" +
                  (f" (showing 1 of {len(uploaded_files)} files)" if is_batch else ""),
            xaxis_title="X-axis",
            yaxis_title="Intensity",
            xaxis=dict(title_font=dict(size=22, color="black"), tickfont=dict(size=20, color="black")),
            yaxis=dict(title_font=dict(size=22, color="black"), tickfont=dict(size=20, color="black")),
            height=600,
            hovermode='x unified',
            font=dict(size=20, color="black"),
            title_font=dict(size=22, color="black"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=18, color="black"),
            )
        )

        st.plotly_chart(fig)

        if is_batch:
            st.select_slider(
                "Preview file",
                options=[f.name for f in uploaded_files],
                key="reduce_preview_file_name",
            )
