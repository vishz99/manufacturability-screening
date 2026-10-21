"""
Subprocess worker: everything in this file that touches occwl/pythonocc
for an arbitrary, untrusted STEP file runs in a separate process from
the caller. A malformed STEP file can segfault pythonocc outright (not
just raise a Python exception), which would otherwise kill the whole
app/kernel. Isolating it in a subprocess means a crash here just makes
this process exit with a nonzero code, which the caller can detect and
report as a clean error instead.

Invoked as:
    <brepnet python> _extract_worker.py <step_path> <output_dir>

On success, writes <output_dir>/part.npz (BRepNet's extracted
face/edge/coedge features, PLUS a combined per-face tessellation:
tess_verts, tess_tris, tess_face_of_tri) and <output_dir>/adjacency.json
(face-index pairs that share an edge, indices in the SAME canonical
face order as everything else -- occwl's raw Compound.faces() order,
which is also the order BRepNet's extraction pipeline writes its npz
rows in; see src/data/mfcad_loader.py and
notebooks/verify_extraction_face_order.py for why that equivalence
holds). Then prints one JSON line to stdout:
    {"ok": true, "num_faces": N}

On failure (unreadable file, non-manifold/open/duplicate-coedge body,
or any other exception), prints:
    {"ok": false, "error": "<reason>"}
and exits with status 1. Never raises past this script's own
try/except -- the caller distinguishes success/failure by exit code and
by parsing the last stdout line as JSON, not by exceptions crossing the
process boundary.
"""

import json
import shutil
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))

FEATURE_LIST_PATH = BREPNET_DIR / "feature_lists" / "all.json"


def fail(reason: str) -> None:
    print(json.dumps({"ok": False, "error": reason}))
    sys.exit(1)


def main():
    if len(sys.argv) != 3:
        fail("usage: _extract_worker.py <step_path> <output_dir>")

    step_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not step_path.exists():
        fail(f"STEP file not found: {step_path}")

    try:
        from occwl.compound import Compound
        from occwl.entity_mapper import EntityMapper
        from occwl.graph import face_adjacency
    except Exception as e:
        fail(f"failed to import occwl: {e}")

    # Step 1: can occwl even read this file and find faces in it?
    try:
        compound = Compound.load_from_step(str(step_path))
        faces = list(compound.faces())
    except Exception as e:
        fail(f"occwl could not read the STEP file: {e}\n{traceback.format_exc()}")

    if len(faces) == 0:
        fail("STEP file contains no faces")

    # Step 2: extract BRepNet's per-face/edge/coedge features. This uses
    # the same isolated-single-file-directory pattern as
    # scripts/extract_features.py, since extract_brepnet_data_from_step
    # globs **/*.step under whatever directory it's given.
    try:
        from pipeline.extract_brepnet_data_from_step import extract_brepnet_data_from_step
    except Exception as e:
        fail(f"failed to import BRepNet extraction pipeline: {e}")

    isolated_input_dir = output_dir / "isolated_input"
    isolated_input_dir.mkdir(parents=True, exist_ok=True)
    isolated_step_path = isolated_input_dir / step_path.name
    shutil.copy(step_path, isolated_step_path)

    npz_output_dir = output_dir / "npz_out"
    npz_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        extract_brepnet_data_from_step(
            isolated_input_dir,
            npz_output_dir,
            feature_list_path=FEATURE_LIST_PATH,
            force_regeneration=True,
        )
    except Exception as e:
        fail(f"BRepNet extraction raised an exception: {e}\n{traceback.format_exc()}")

    extracted_npz_path = npz_output_dir / f"{isolated_step_path.stem}.npz"
    if not extracted_npz_path.exists():
        # extract_brepnet_data_from_step does not raise on a non-manifold,
        # unclosed, or duplicate-coedge body -- it prints a warning and
        # simply doesn't write the npz. Absence of the file is the real
        # success/failure signal, not the absence of an exception.
        fail(
            "BRepNet extraction did not produce output -- the body is likely "
            "non-manifold, not closed, or has a coedge used in multiple loops "
            "(see BRepNetExtractor.check_manifold/check_closed/check_unique_coedges)"
        )

    import numpy as np

    extracted = dict(np.load(extracted_npz_path))

    if extracted["face_features"].shape[0] != len(faces):
        fail(
            f"face count mismatch: extraction produced {extracted['face_features'].shape[0]} "
            f"face rows but occwl found {len(faces)} faces in the same file"
        )

    # Step 3: tessellate every face for the 3D viewer, in the SAME
    # canonical order as the extracted features (occwl's raw
    # Compound.faces() traversal order). One combined mesh with a
    # per-triangle face-index array, so the viewer can color by face
    # without building one trace per face.
    try:
        # Must triangulate the whole shape first -- Face.get_triangles()
        # just reads back an already-computed mesh (BRep_Tool.Triangulation),
        # it does not compute one itself.
        compound.triangulate_all_faces()

        all_verts = []
        all_tris = []
        tri_face_of = []
        vert_offset = 0
        for face_idx, face in enumerate(faces):
            fverts, ftris = face.get_triangles()
            if fverts is None or len(fverts) == 0:
                continue
            all_verts.append(fverts)
            all_tris.append(ftris + vert_offset)
            tri_face_of.extend([face_idx] * len(ftris))
            vert_offset += len(fverts)

        tess_verts = np.concatenate(all_verts, axis=0) if all_verts else np.zeros((0, 3), dtype=np.float32)
        tess_tris = np.concatenate(all_tris, axis=0) if all_tris else np.zeros((0, 3), dtype=np.int32)
        tess_face_of_tri = np.array(tri_face_of, dtype=np.int32)
    except Exception as e:
        fail(f"tessellation failed: {e}\n{traceback.format_exc()}")

    # Step 4: face adjacency graph, remapped from occwl's own EntityMapper
    # index (whatever internal order that assigns) to OUR canonical face
    # order, via actual face-object identity -- not assumed to match.
    try:
        mapper = EntityMapper(compound)
        occwl_idx_to_canonical = {mapper.face_index(f.topods_shape()): i for i, f in enumerate(faces)}
        graph = face_adjacency(compound)
        edges = []
        if graph is not None:
            for u, v in graph.edges():
                cu = occwl_idx_to_canonical.get(u)
                cv = occwl_idx_to_canonical.get(v)
                if cu is not None and cv is not None:
                    edges.append((cu, cv))
    except Exception as e:
        fail(f"face adjacency graph failed: {e}\n{traceback.format_exc()}")

    extracted["tess_verts"] = tess_verts
    extracted["tess_tris"] = tess_tris
    extracted["tess_face_of_tri"] = tess_face_of_tri
    np.savez(output_dir / "part.npz", **extracted)

    (output_dir / "adjacency.json").write_text(json.dumps({"num_faces": len(faces), "edges": edges}))

    print(json.dumps({"ok": True, "num_faces": len(faces)}))


if __name__ == "__main__":
    main()
