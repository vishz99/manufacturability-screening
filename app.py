"""
Streamlit app: upload a STEP file (or pick a validation-split part),
see the accepted model's per-face predictions as a colored 3D view,
the feature summary table, and rule-based manufacturability flags.

Layout is a single row of three columns (viewer, summary, flags) kept
to a fixed, modest height each, so the whole page fits one screen
without scrolling -- the legend and disclaimer are collapsed into
expanders/captions rather than stacked below.

Proof-of-concept grade: see docs/project-summary.md for the real
accuracy (60.35% validation) and its limitations before relying on this
for anything beyond a demo.

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe -m streamlit run app.py
"""

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.mfcad_loader import FEAT_NAMES  # noqa: E402
from inference.predict_part import PartAnalysisError  # noqa: E402
from screening.rules import DISCLAIMER, DEFAULT_CONFIDENCE_THRESHOLD  # noqa: E402
from ui.analysis import analyse_part  # noqa: E402
from ui.viewer import build_mesh_figure, legend_html  # noqa: E402

STEP_DIR = PROJECT_ROOT / "data" / "MFCAD" / "dataset" / "step"
SPLIT_PATH = PROJECT_ROOT / "outputs" / "mfcad_train_val_split.json"

TABLE_HEIGHT = 300  # fixed, so neither table grows the page past one screen

st.set_page_config(page_title="Manufacturability Screening", layout="wide")
st.markdown(
    "##### Manufacturability Screening &nbsp;"
    "<span style='font-size:0.75em;color:gray'>proof-of-concept, 60.35% face accuracy -- see docs/project-summary.md</span>",
    unsafe_allow_html=True,
)


@st.cache_data
def _val_stems():
    split = json.loads(SPLIT_PATH.read_text())
    return sorted(split["val_set"])


with st.sidebar:
    st.header("Part selection")
    source = st.radio("Source", ["Validation part", "Upload STEP file"])

    step_path = None
    uploaded_bytes = None
    color_source = "Prediction"

    if source == "Validation part":
        stem = st.selectbox("Validation part", _val_stems())
        step_path = STEP_DIR / f"{stem}.step"
        color_source = st.radio("Colour by", ["Prediction", "Ground truth"])
    else:
        uploaded = st.file_uploader("STEP file", type=["step", "stp"])
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()

    confidence_threshold = st.slider(
        "Confidence threshold for flags", min_value=0.3, max_value=0.9,
        value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05,
    )

    with st.expander("Class colour legend"):
        st.markdown(legend_html(FEAT_NAMES), unsafe_allow_html=True)


def _run_analysis(path: Path, threshold: float):
    try:
        return analyse_part(path, confidence_threshold=threshold), None
    except PartAnalysisError as e:
        return None, str(e)
    except Exception as e:  # noqa: BLE001 -- surface any other failure as a clean error, never a crash
        return None, f"Unexpected error: {e}"


analysis = None
error = None

if source == "Validation part" and step_path is not None:
    analysis, error = _run_analysis(step_path, confidence_threshold)
elif source == "Upload STEP file" and uploaded_bytes is not None:
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
        tmp.write(uploaded_bytes)
        tmp_path = Path(tmp.name)
    analysis, error = _run_analysis(tmp_path, confidence_threshold)
    tmp_path.unlink(missing_ok=True)
elif source == "Upload STEP file":
    st.info("Upload a STEP file to analyze it.")

if error:
    st.error(error)

if analysis is not None:
    col_viewer, col_summary, col_flags = st.columns([3, 2, 2])

    with col_viewer:
        if source == "Validation part" and color_source == "Ground truth" and analysis.ground_truth is not None:
            class_per_face = analysis.ground_truth.true_labels
            confidence_per_face = None
            title = "Ground truth"
        else:
            class_per_face = [fp.predicted_class for fp in analysis.prediction.predictions]
            confidence_per_face = [fp.confidence for fp in analysis.prediction.predictions]
            title = "Prediction"

        fig = build_mesh_figure(
            tess_verts=analysis.prediction.tess_verts,
            tess_tris=analysis.prediction.tess_tris,
            tess_face_of_tri=analysis.prediction.tess_face_of_tri,
            class_per_face=class_per_face,
            feat_names=FEAT_NAMES,
            confidence_per_face=confidence_per_face,
            confidence_threshold=confidence_threshold,
            title=title,
        )
        st.plotly_chart(fig, use_container_width=True)
        faces_line = f"Faces: {analysis.prediction.num_faces}"
        if analysis.accuracy is not None:
            faces_line += f"   |   Accuracy vs. ground truth: {analysis.accuracy:.4f}"
        st.caption(faces_line)

    with col_summary:
        st.markdown("**Predicted features**")
        st.dataframe(
            [
                {
                    "class": s.class_name,
                    "faces": s.face_count,
                    "inst.": s.instance_count if s.instance_count is not None else "n/a",
                    "conf.": f"{s.mean_confidence:.2f}" if s.mean_confidence is not None else "n/a",
                    "reliability": s.reliability,
                }
                for s in analysis.predicted_summary
            ],
            use_container_width=True,
            hide_index=True,
            height=TABLE_HEIGHT,
        )

    with col_flags:
        st.markdown("**Manufacturability flags**")
        if analysis.flags:
            st.dataframe(
                [
                    {
                        "severity": f.severity,
                        "class": f.class_name,
                        "flag": f.message,
                        "reliability": f.reliability,
                    }
                    for f in sorted(analysis.flags, key=lambda f: (f.severity != "review", f.class_name))
                ],
                use_container_width=True,
                hide_index=True,
                height=TABLE_HEIGHT,
            )
        else:
            st.write("No flags for the current confidence threshold.")
        st.caption(DISCLAIMER)
