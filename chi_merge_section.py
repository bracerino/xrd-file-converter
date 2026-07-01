"""Chi-tilt .xy merging tool (Maximum-Projection method).

Merge a series of powder-diffraction .xy patterns measured at different sample
inclinations (chi tilts) into a single 1D pattern that preserves all diffraction
peaks. A plain average dilutes single-tilt spots into the background ~1/N; taking
the per-2theta maximum across tilts keeps each crystallite's full peak intensity.

Exposed as ``run_chi_merge_section()`` for the Streamlit app.
"""

import io
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import (
    minimum_filter1d,
    maximum_filter1d,
    uniform_filter1d,
)


FONT_SIZE   = 24
HOVER_FONT  = 22
LEGEND_FONT = 18
AXIS_FONT   = 22


def _style(fig, x_title, y_title, legend=True, height=600):
    """Apply consistent, large-font styling to a plotly figure."""
    fig.update_layout(
        font=dict(size=FONT_SIZE),
        hoverlabel=dict(font_size=HOVER_FONT),
        height=height,
        margin=dict(l=80, r=20, t=30, b=70),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=AXIS_FONT)),
            tickfont=dict(size=LEGEND_FONT),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=AXIS_FONT)),
            tickfont=dict(size=LEGEND_FONT),
        ),
    )
    if legend:
        fig.update_layout(legend=dict(font=dict(size=LEGEND_FONT)))
    return fig


# --------------------------------------------------------------------------- #
#  Parsing helpers
# --------------------------------------------------------------------------- #
def _parse_chi_from_name(name: str):
    """Extract the chi tilt value encoded in a filename, e.g. ``...Chi=12.00.xy``.

    Only an explicit ``chi`` token is recognised so that ordinary numbered files
    (e.g. ``scan_003.xy``) are treated as generic data, not as an angle.
    Returns a float, or ``None`` if no chi token is found.
    """
    m = re.search(r"chi\s*[=_-]?\s*(-?\d+(?:\.\d+)?)", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _parse_xy(uploaded_file) -> np.ndarray:
    """Read a two-column ASCII .xy file into an (N, 2) array (2theta, intensity).

    Comment lines (starting with '#', ';', '%') and non-numeric header lines are
    skipped. Accepts whitespace- or comma-separated columns.
    """
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    try:
        text = raw.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        text = raw.decode("latin-1", errors="replace")

    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] in "#;%":
            continue
        parts = re.split(r"[,\s]+", s)
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        rows.append((x, y))

    if not rows:
        return np.empty((0, 2))
    return np.asarray(rows, dtype=float)


# --------------------------------------------------------------------------- #
#  Core algorithm
# --------------------------------------------------------------------------- #
def _background(y: np.ndarray, win: int = 81) -> np.ndarray:
    """Smooth background estimate: rolling-min -> rolling-max -> moving-average."""
    win = max(3, min(win, len(y) if len(y) % 2 else len(y) - 1))
    b = minimum_filter1d(y, win, mode="nearest")
    b = maximum_filter1d(b, win, mode="nearest")
    return uniform_filter1d(b, win, mode="nearest")


def build_merge(x, Y, bg_win=81):
    """Compute all merge variants from a stacked (N_chi, N_2theta) matrix.

    Returns a dict with keys: maxproj, p95, average, bg_max, common_floor.
    """
    out = {}
    out["maxproj"] = Y.max(axis=0)
    out["p95"]     = np.percentile(Y, 95, axis=0)
    out["average"] = Y.mean(axis=0)

    # background-subtracted max + smooth common floor added back
    bg_stack = np.array([_background(yi, bg_win) for yi in Y])
    Y_bs = np.clip(Y - bg_stack, 0, None)
    common_floor = bg_stack.min(axis=0)
    out["bg_max"]       = Y_bs.max(axis=0) + common_floor
    out["bg_max_only"]  = Y_bs.max(axis=0)            # for peak picking
    out["common_floor"] = common_floor
    return out


def detect_peaks(x, peaks_only, sg_win=11, sg_order=3, prom_mult=4.0, distance=4):
    """Peak detection on a background-subtracted profile.

    Returns (peak_2theta, peak_intensity, sigma).
    """
    n = len(peaks_only)
    win = sg_win if sg_win % 2 else sg_win + 1
    if win >= n:
        win = n - 1 if (n - 1) % 2 else n - 2
    if win >= 5 and win > sg_order:
        sig = savgol_filter(peaks_only, win, sg_order)
    else:
        sig = peaks_only.copy()

    diffs = np.abs(np.diff(sig))
    sigma = np.median(diffs) * 1.4826 if diffs.size else 0.0
    prominence = prom_mult * sigma if sigma > 0 else None

    idx, _ = find_peaks(sig, prominence=prominence, distance=max(1, distance))
    return x[idx], peaks_only[idx], sigma


# --------------------------------------------------------------------------- #
#  Grid handling
# --------------------------------------------------------------------------- #
def _stack_patterns(patterns, ref_x=None):
    """Stack a list of (x, y) arrays onto a common 2theta grid.

    If all share an identical grid, that grid is used as-is. Otherwise every
    pattern is linearly interpolated onto a common grid (the overlap of all
    ranges, sampled at the finest median step).

    Returns (x, Y, regridded_flag).
    """
    xs = [p[:, 0] for p in patterns]
    ys = [p[:, 1] for p in patterns]

    same = all(xs[0].shape == xi.shape for xi in xs) and \
        all(np.max(np.abs(xi - xs[0])) == 0 for xi in xs)
    if same:
        return xs[0], np.array(ys), False

    lo = max(xi.min() for xi in xs)
    hi = min(xi.max() for xi in xs)
    step = np.median([np.median(np.diff(xi)) for xi in xs])
    if not np.isfinite(step) or step <= 0:
        step = (hi - lo) / max(len(xs[0]) - 1, 1)
    grid = np.arange(lo, hi + step / 2, step)
    Y = np.array([np.interp(grid, xi, yi) for xi, yi in zip(xs, ys)])
    return grid, Y, True


# --------------------------------------------------------------------------- #
#  Export helpers
# --------------------------------------------------------------------------- #
def _xy_text(x, y, header):
    body = "\n".join(f"{xi:.6f}  {yi:.4f}" for xi, yi in zip(x, y))
    return f"# {header}\n{body}\n"


def _make_zip(x, variants, peaks_tt, peaks_int, base_name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_name}_merged_maxproj.xy",
                    _xy_text(x, variants["maxproj"], "2Theta  Intensity (max projection)"))
        zf.writestr(f"{base_name}_merged_p95.xy",
                    _xy_text(x, variants["p95"], "2Theta  Intensity (95th percentile)"))
        zf.writestr(f"{base_name}_merged_bgmax.xy",
                    _xy_text(x, variants["bg_max"], "2Theta  Intensity (background-subtracted max)"))
        zf.writestr(f"{base_name}_merged_average.xy",
                    _xy_text(x, variants["average"], "2Theta  Intensity (average, reference only)"))
        peak_body = "# 2Theta  Intensity\n" + "\n".join(
            f"{t:.6f}  {i:.4f}" for t, i in zip(peaks_tt, peaks_int))
        zf.writestr(f"{base_name}_peaks.txt", peak_body + "\n")
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  Streamlit entry point
# --------------------------------------------------------------------------- #
def run_chi_merge_section():
    st.markdown("### 🧬 .xy Pattern Merge (Maximum Projection)")
    st.info(
        "Upload a series of two-column **.xy** patterns (`X  Intensity`) that you want "
        "to merge into a single pattern. For coarse-grained / spotty samples a plain "
        "average dilutes reflections that appear in only one scan into the background. "
        "This tool instead takes the **per-point maximum across all patterns**, "
        "preserving every peak.\n\n"
        "This works on **any** mergeable .xy data. If the filenames encode a sample "
        "inclination (**χ tilt**, e.g. `...Chi=12.00.xy`) it is read automatically and "
        "used for labelling / ordering — otherwise the upload order is used.\n\n"
        "⚠️ **Max-projection intensities are not quantitative** (a peak's height "
        "reflects how many grains happened to diffract, not phase fraction). Use the "
        "merged pattern for peak finding / phase ID only — for Rietveld / quantitative "
        "refinement use the raw data."
    )

    uploaded = st.file_uploader(
        "Upload .xy files to merge",
        type=["xy", "txt", "dat", "asc"],
        accept_multiple_files=True,
        key="chi_merge_upload",
    )

    if not uploaded:
        return

    if len(uploaded) < 2:
        st.warning("Please upload at least two .xy files to merge.")
        return

    # ---- parse ---------------------------------------------------------- #
    parsed = []
    for f in uploaded:
        arr = _parse_xy(f)
        if arr.shape[0] == 0:
            st.error(f"No numeric two-column data found in **{f.name}** — skipped.")
            continue
        # sort by 2theta to be safe
        arr = arr[np.argsort(arr[:, 0])]
        parsed.append((f.name, _parse_chi_from_name(f.name), arr))

    if len(parsed) < 2:
        st.error("Fewer than two files could be parsed.")
        return

    # order by chi tilt when available, otherwise keep upload order
    if all(p[1] is not None for p in parsed):
        parsed.sort(key=lambda p: p[1])

    names    = [p[0] for p in parsed]
    chi_vals = [p[1] for p in parsed]
    patterns = [p[2] for p in parsed]

    base_name = _common_prefix([n.rsplit(".", 1)[0] for n in names]) or "merged"
    base_name = base_name.rstrip("_-. ") or "merged"

    # ---- sidebar settings ---------------------------------------------- #
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 🧬 Merge Settings")
        normalize = st.checkbox(
            "Normalise each pattern (by total integrated intensity)",
            value=False,
            help="Use when exposure / counting time varies per tilt so the patterns "
                 "are on a comparable scale before merging.",
        )
        bg_win = st.slider("Background window (points)", 11, 201, 81, step=2,
                           help="Window for the rolling min→max→average background estimate.")
        st.markdown("**Peak detection**")
        prom_mult = st.slider("Prominence threshold (× σ)", 1.0, 10.0, 4.0, step=0.5)
        distance  = st.slider("Min. peak separation (points)", 1, 20, 4)
        sg_win    = st.slider("Smoothing window (Savitzky–Golay)", 5, 51, 11, step=2)

    # ---- stack onto common grid ---------------------------------------- #
    x, Y, regridded = _stack_patterns(patterns)

    if normalize:
        totals = Y.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        ref = np.median(totals)
        Y = Y / totals * ref

    # χ tilt is used only when *every* uploaded file encodes one
    has_chi   = all(c is not None for c in chi_vals)
    chi_known = [c for c in chi_vals if c is not None]

    # ---- metrics -------------------------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patterns" + (" (χ tilts)" if has_chi else ""), len(patterns))
    c2.metric("X points", len(x))
    c3.metric("X range", f"{x.min():.2f} – {x.max():.2f}")
    if has_chi:
        c4.metric("χ range", f"{min(chi_known):.1f}° – {max(chi_known):.1f}°")
    else:
        c4.metric("Ordering", "upload order")

    if regridded:
        st.warning(
            "⚠️ The uploaded files do **not** share an identical X grid — patterns "
            "were linearly **interpolated** onto a common grid before merging.")
    else:
        st.success("✅ All files share an identical X grid (no regridding needed).")

    with st.expander("📋 Uploaded files", expanded=False):
        files_tbl = {"File": names}
        if has_chi:
            files_tbl["χ tilt (°)"] = [f"{c:.2f}" for c in chi_vals]
        files_tbl["Points"] = [p.shape[0] for p in patterns]
        st.dataframe(pd.DataFrame(files_tbl), width="stretch", hide_index=True)

    # ---- compute merges ------------------------------------------------- #
    variants = build_merge(x, Y, bg_win=bg_win)
    peaks_tt, peaks_int, sigma = detect_peaks(
        x, variants["bg_max_only"],
        sg_win=sg_win, prom_mult=prom_mult, distance=distance,
    )

    method_map = {
        "Max projection":            ("maxproj", "black"),
        "95th percentile":           ("p95", "royalblue"),
        "Background-subtracted max": ("bg_max", "green"),
        "Average":                   ("average", "firebrick"),
    }

    # ---- plotting ------------------------------------------------------- #
    heatmap_label = "🌡️ χ–2θ heatmap" if has_chi else "🌡️ Heatmap"
    tab_overlay, tab_methods, tab_heatmap, tab_peaks, tab_dl = st.tabs(
        ["📈 All patterns + merged", "🔬 Merge methods", heatmap_label,
         "📍 Peaks", "💾 Download"]
    )

    with tab_overlay:
        st.markdown("#### All input patterns with the merged result")
        oc1, oc2, oc3 = st.columns([2, 1, 1])
        merged_choice = oc1.selectbox(
            "Merged pattern to overlay",
            list(method_map.keys()),
            index=0,
            key="chi_merge_overlay_choice",
        )
        n_pat = len(Y)
        every = oc2.number_input(
            "Show every Nth pattern",
            min_value=1, max_value=max(1, n_pat), value=1, step=1,
            key="chi_merge_every",
            help="1 = show all patterns, 2 = every second, 3 = every third, …",
        )
        offset_on = oc3.checkbox("Stack with vertical offset", value=False,
                                 key="chi_merge_offset")
        mkey, mcolor = method_map[merged_choice]

        shown_idx = list(range(0, n_pat, int(every)))
        if int(every) > 1:
            st.caption(f"Showing {len(shown_idx)} of {n_pat} patterns "
                       f"(every {int(every)}). The merge still uses **all** patterns.")

        fig = go.Figure()
        span = float(np.nanmax(Y) - np.nanmin(Y)) or 1.0
        step = 0.15 * span
        for plot_i, i in enumerate(shown_idx):
            yi, c, nm = Y[i], chi_vals[i], names[i]
            off = plot_i * step if offset_on else 0.0
            label = f"χ={c:.2f}°" if c is not None else nm
            fig.add_trace(go.Scatter(
                x=x, y=yi + off, mode="lines", name=label,
                line=dict(width=1),
                opacity=0.7,
            ))
        fig.add_trace(go.Scatter(
            x=x, y=variants[mkey], mode="lines",
            name=f"MERGED ({merged_choice})",
            line=dict(width=3, color=mcolor),
        ))
        _style(fig, "2θ / X", "Intensity" + (" (offset)" if offset_on else ""))
        st.plotly_chart(fig, width="stretch")

    with tab_methods:
        st.markdown("#### Comparison of merge methods")
        show = st.multiselect(
            "Methods to show",
            list(method_map.keys()),
            default=["Max projection", "Background-subtracted max", "Average"],
            key="chi_merge_methods_show",
        )
        fig2 = go.Figure()
        for label in show:
            key, color = method_map[label]
            fig2.add_trace(go.Scatter(
                x=x, y=variants[key], mode="lines", name=label,
                line=dict(width=2.5, color=color),
            ))
        _style(fig2, "2θ / X", "Intensity")
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "**Average** is shown for reference only — it demonstrates how single-tilt "
            "reflections get diluted. **95th percentile** has a lower baseline than max "
            "but discards reflections present in only 1–2 frames.")

    with tab_heatmap:
        st.markdown("#### " + ("χ–2θ intensity heatmap" if has_chi
                               else "Pattern intensity heatmap"))
        st.caption("The most useful diagnostic for confirming spotty / single-pattern "
                   "reflections — each bright streak that appears in only one row is a "
                   "spot that averaging would destroy.")
        y_axis = [f"{c:.2f}" for c in chi_vals] if has_chi else names
        fig3 = go.Figure(data=go.Heatmap(
            z=Y, x=x, y=y_axis, colorscale="Viridis",
            colorbar=dict(title=dict(text="Intensity", font=dict(size=AXIS_FONT)),
                          tickfont=dict(size=LEGEND_FONT)),
        ))
        _style(fig3, "2θ / X", "χ tilt (°)" if has_chi else "File", legend=False)
        st.plotly_chart(fig3, width="stretch")

    with tab_peaks:
        st.markdown(f"#### Detected peaks ({len(peaks_tt)} found)")
        st.caption(
            f"Noise estimate σ ≈ {sigma:.3g}; peaks require prominence ≥ "
            f"{prom_mult:.1f}·σ on the background-subtracted max profile. "
            "Auto-detected peaks can include noise spikes or Kα₂ shoulders — review "
            "before use.")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=x, y=variants["bg_max"], mode="lines",
            name="Background-subtracted max", line=dict(width=1.5, color="gray"),
        ))
        if len(peaks_tt):
            # sticks at detected peak positions
            base = float(np.min(variants["bg_max"]))
            for t, h in zip(peaks_tt, peaks_int):
                fig4.add_trace(go.Scatter(
                    x=[t, t], y=[base, base + h], mode="lines",
                    line=dict(width=1.5, color="red"),
                    showlegend=False, hoverinfo="skip",
                ))
            fig4.add_trace(go.Scatter(
                x=peaks_tt, y=base + peaks_int, mode="markers",
                name="Peaks", marker=dict(color="red", size=10, symbol="triangle-down"),
            ))
        _style(fig4, "2θ / X", "Intensity")
        st.plotly_chart(fig4, width="stretch")

        if len(peaks_tt):
            peaks_df = pd.DataFrame({
                "2θ / X": np.round(peaks_tt, 4),
                "Intensity": np.round(peaks_int, 2),
            })
            st.dataframe(peaks_df, width="stretch", hide_index=True)
        else:
            st.info("No peaks found with the current settings — try lowering the "
                    "prominence threshold.")

    with tab_dl:
        st.markdown("#### Download merged data")
        d1, d2 = st.columns(2)
        d1.download_button(
            "💾 Merged (max projection) .xy",
            data=_xy_text(x, variants["maxproj"], "2Theta  Intensity (max projection)"),
            file_name=f"{base_name}_merged_maxproj.xy",
            mime="text/plain",
            type="primary",
        )
        d2.download_button(
            "💾 Background-subtracted max .xy",
            data=_xy_text(x, variants["bg_max"], "2Theta  Intensity (background-subtracted max)"),
            file_name=f"{base_name}_merged_bgmax.xy",
            mime="text/plain",
        )

        peak_body = "# 2Theta  Intensity\n" + "\n".join(
            f"{t:.6f}  {i:.4f}" for t, i in zip(peaks_tt, peaks_int))
        st.download_button(
            "📍 Peak list (.txt)",
            data=peak_body + "\n",
            file_name=f"{base_name}_peaks.txt",
            mime="text/plain",
        )

        st.markdown("---")
        st.markdown("**Full archive** — all variants + peak list")
        zip_bytes = _make_zip(x, variants, peaks_tt, peaks_int, base_name)
        st.download_button(
            "📦 Download ZIP (all outputs)",
            data=zip_bytes,
            file_name=f"{base_name}_merge_outputs.zip",
            mime="application/zip",
            type="primary",
        )


def _common_prefix(names: list) -> str:
    if not names:
        return ""
    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
