"""
Interactive 3D view of a part's tessellation, colored per face by
predicted (or true) class, built with plotly's Mesh3d (one combined
mesh, one color per triangle via `facecolor` -- not one trace per face,
which doesn't scale well in a browser for parts with many faces).

Tessellation data (tess_verts/tess_tris/tess_face_of_tri) comes from
src/inference/predict_part.py's PartPrediction, produced inside the
extraction subprocess via occwl's own triangulation
(TriangulatorMixin.get_triangles(), see src/inference/_extract_worker.py)
-- not a new meshing implementation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

# Colorblind-friendly categorical palette: Okabe-Ito's 8 colors
# (https://jfly.uni-koeln.de/color/) followed by 8 from Paul Tol's
# "muted" qualitative palette (https://sronpersonalpages.nl/~pault/),
# both established colorblind-safe categorical sets. 16 entries, one
# per MFCAD class, in FEAT_NAMES order.
CLASS_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
    "#44AA99", "#882255", "#AA4499", "#DDCC77",
    "#332288", "#117733", "#88CCEE", "#661100",
]
LOW_CONFIDENCE_COLOR = "#BBBBBB"  # neutral grey, same for every class


def _triangle_colors(
    face_of_tri: np.ndarray,
    class_per_face: Sequence[int],
    confidence_per_face: Optional[Sequence[float]],
    confidence_threshold: float,
) -> List[str]:
    colors = []
    for face_idx in face_of_tri:
        cls = class_per_face[face_idx]
        if confidence_per_face is not None and confidence_per_face[face_idx] < confidence_threshold:
            colors.append(LOW_CONFIDENCE_COLOR)
        else:
            colors.append(CLASS_COLORS[cls % len(CLASS_COLORS)])
    return colors


def build_mesh_figure(
    tess_verts: np.ndarray,
    tess_tris: np.ndarray,
    tess_face_of_tri: np.ndarray,
    class_per_face: Sequence[int],
    feat_names: Sequence[str],
    confidence_per_face: Optional[Sequence[float]] = None,
    confidence_threshold: float = 0.5,
    title: str = "",
    height: int = 420,
) -> go.Figure:
    """Builds one Mesh3d figure, triangles colored by class_per_face
    (greyed out below confidence_threshold if confidence_per_face is
    given -- pass None, e.g. for ground truth, to disable greying).
    height is capped by default (420px) so the figure fits one screen
    alongside the summary table and flags, rather than pushing the page
    past the viewport."""
    if len(tess_tris) == 0:
        fig = go.Figure()
        fig.update_layout(title=title or "No tessellation available", height=height)
        return fig

    colors = _triangle_colors(tess_face_of_tri, class_per_face, confidence_per_face, confidence_threshold)

    mesh = go.Mesh3d(
        x=tess_verts[:, 0], y=tess_verts[:, 1], z=tess_verts[:, 2],
        i=tess_tris[:, 0], j=tess_tris[:, 1], k=tess_tris[:, 2],
        facecolor=colors,
        flatshading=True,
        hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1),
    )
    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=dict(text=title, y=0.98, pad=dict(t=0)),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=24, b=0),
        height=height,
    )
    return fig


def legend_html(feat_names: Sequence[str]) -> str:
    """A small HTML color-swatch legend, for use with st.markdown(unsafe_allow_html=True)."""
    rows = []
    for name, color in zip(feat_names, CLASS_COLORS):
        rows.append(
            f'<span style="display:inline-block;width:12px;height:12px;background:{color};'
            f'margin-right:6px;border:1px solid #333;"></span>{name}'
        )
    rows.append(
        f'<span style="display:inline-block;width:12px;height:12px;background:{LOW_CONFIDENCE_COLOR};'
        f'margin-right:6px;border:1px solid #333;"></span>below confidence threshold'
    )
    return "<br>".join(rows)
