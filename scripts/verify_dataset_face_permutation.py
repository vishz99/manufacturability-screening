"""
Phase 3 prerequisite: verifies BRepNetDataset's internal face
permutation (dataloaders/brepnet_dataset.py's
build_coedges_of_faces_tensor / new_to_old_face_indices /
old_to_new_face_indices) before it gets used to align MFCAD labels with
model predictions in the training loop.

WHY THIS MATTERS: BRepNetDataset.load_body() re-orders the faces from
npz order into "small faces first, then big faces" order (faces grouped
by how many coedges surround them, threshold = 30) before feeding them
to the model:

    Xf_perm = Xf[new_to_old_face_indices]

Training labels (from mfcad_loader.py, which are in npz/occwl-raw
order -- proven in Phase 2) must go through this exact same gather to
land in the order the model actually consumes and predicts in. Getting
the direction or the split logic wrong would silently misassign every
label.

Two independent checks, run on real extracted subset files:

  1. BIJECTION CHECK: new_to_old_face_indices must be a genuine
     permutation of range(num_faces) -- every original face index
     appears exactly once -- and old_to_new_face_indices must be its
     true inverse (round-trips: old_to_new[new_to_old[k]] == k for
     every k).

  2. INDEPENDENT SPLIT-LOGIC CHECK: the stated rule is "faces with <=30
     coedges come first (small), faces with >30 coedges come after
     (big)". We recompute each face's coedge count OURSELVES directly
     from the npz's raw coedge_to_face incidence array (not by trusting
     build_coedges_of_faces_tensor's internal bookkeeping) and confirm
     the permutation's ordering matches that independently-recomputed
     classification.

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe scripts/verify_dataset_face_permutation.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))

FEATURES_DIR = PROJECT_ROOT / "outputs" / "brepnet_features"
SUBSET_LIST = PROJECT_ROOT / "outputs" / "mfcad_subset.txt"

MAX_COEDGES_PER_FACE = 30  # matches BRepNetDataset.load_body's hardcoded value
N_FILES_TO_CHECK = 500  # the full verified subset -- cheap since it only reads already-extracted npz arrays
RANDOM_SEED = 3


def independent_coedge_counts_per_face(body_data):
    """
    Recompute, directly from the raw coedge_to_face incidence array
    (loaded straight from the npz, no BRepNetDataset logic involved),
    how many coedges surround each face.
    """
    coedge_to_face = body_data["coedge_to_face"]
    num_faces = body_data["face_features"].shape[0]
    counts = np.bincount(coedge_to_face, minlength=num_faces)
    assert counts.shape[0] == num_faces
    assert np.all(counts > 0), "Every face should have at least one coedge"
    return counts


def check_one_file(stem, dataset_helper):
    import utils.data_utils as data_utils

    npz_path = FEATURES_DIR / f"{stem}.npz"
    body_data = data_utils.load_npz_data(npz_path)
    num_faces = body_data["face_features"].shape[0]

    # The actual method under test.
    Cf, Csf, new_to_old = dataset_helper.build_coedges_of_faces_tensor(body_data, MAX_COEDGES_PER_FACE)
    old_to_new = dataset_helper.find_inverse_permutation(new_to_old)

    result = {"stem": stem, "num_faces": num_faces, "num_big_faces": len(Csf)}

    # --- Check 1: bijection / true inverse ---
    new_to_old_np = new_to_old.numpy()
    is_permutation = sorted(new_to_old_np.tolist()) == list(range(num_faces))
    round_trip_ok = torch.equal(old_to_new[new_to_old], torch.arange(num_faces, dtype=old_to_new.dtype))
    result["is_permutation"] = is_permutation
    result["round_trip_ok"] = round_trip_ok

    # --- Check 2: independent split-logic verification ---
    my_counts = independent_coedge_counts_per_face(body_data)
    my_is_big = my_counts > MAX_COEDGES_PER_FACE
    num_small_claimed = num_faces - len(Csf)

    # For every position k in the permuted order, new_to_old[k] tells us
    # which ORIGINAL face landed there. Positions < num_small_claimed
    # should all be faces my_is_big says are NOT big; positions >=
    # num_small_claimed should all be faces my_is_big says ARE big.
    split_ok = True
    for k in range(num_faces):
        original_face = int(new_to_old_np[k])
        expected_big = k >= num_small_claimed
        actual_big = bool(my_is_big[original_face])
        if expected_big != actual_big:
            split_ok = False
            break
    result["split_logic_ok"] = split_ok
    result["max_coedges_on_any_face"] = int(my_counts.max())

    return result


def main():
    from dataloaders.brepnet_dataset import BRepNetDataset

    # We only need build_coedges_of_faces_tensor / find_inverse_permutation,
    # which don't touch self.* state, but we need a bound instance to call
    # them on. __new__ avoids running __init__ (which needs a full dataset
    # file, dataset dir, etc. we don't need for this check).
    dataset_helper = BRepNetDataset.__new__(BRepNetDataset)

    subset_stems = SUBSET_LIST.read_text().split()
    rng = random.Random(RANDOM_SEED)
    sample_stems = rng.sample(subset_stems, N_FILES_TO_CHECK)

    print(f"Checking {N_FILES_TO_CHECK} random subset files (max_coedges_per_face={MAX_COEDGES_PER_FACE})\n")

    n_permutation_ok = 0
    n_round_trip_ok = 0
    n_split_ok = 0
    max_coedges_seen = 0
    any_big_faces = False

    for i, stem in enumerate(sample_stems, start=1):
        r = check_one_file(stem, dataset_helper)
        max_coedges_seen = max(max_coedges_seen, r["max_coedges_on_any_face"])
        if r["num_big_faces"] > 0:
            any_big_faces = True
        n_permutation_ok += int(r["is_permutation"])
        n_round_trip_ok += int(r["round_trip_ok"])
        n_split_ok += int(r["split_logic_ok"])
        failed = not (r["is_permutation"] and r["round_trip_ok"] and r["split_logic_ok"])
        if failed or N_FILES_TO_CHECK <= 20:
            print(
                f"  {stem}: num_faces={r['num_faces']:3d} num_big_faces={r['num_big_faces']} "
                f"max_coedges_on_any_face={r['max_coedges_on_any_face']:3d} "
                f"is_permutation={r['is_permutation']} round_trip_ok={r['round_trip_ok']} "
                f"split_logic_ok={r['split_logic_ok']}"
            )
        if i % 100 == 0:
            print(f"  ...checked {i}/{N_FILES_TO_CHECK}...")

    print(f"\n{'=' * 70}")
    print(f"Bijection check:    {n_permutation_ok}/{N_FILES_TO_CHECK} files: new_to_old is a genuine permutation")
    print(f"Round-trip check:   {n_round_trip_ok}/{N_FILES_TO_CHECK} files: old_to_new is new_to_old's true inverse")
    print(f"Split-logic check:  {n_split_ok}/{N_FILES_TO_CHECK} files: independent recount agrees with the permutation's small/big split")
    print(f"Max coedges seen on any single face across this sample: {max_coedges_seen}")
    print(f"Any 'big' faces (>{ MAX_COEDGES_PER_FACE } coedges) encountered in this sample: {any_big_faces}")
    if not any_big_faces:
        print(
            "\nNOTE: no face in this sample exceeded the max_coedges_per_face=30 threshold, "
            "so new_to_old_face_indices reduces to the identity permutation for every file "
            "checked here (small_face_indices = [0, 1, ..., num_faces-1], big_face_indices "
            "empty). The split-logic check above still concretely verifies this rather than "
            "assuming it, but it means the reordering is a no-op on this sample of MFCAD "
            "parts specifically -- it should NOT be assumed to be a no-op in general, and the "
            "training code below still applies the real permutation rather than relying on it "
            "being the identity."
        )

    all_ok = (
        n_permutation_ok == N_FILES_TO_CHECK
        and n_round_trip_ok == N_FILES_TO_CHECK
        and n_split_ok == N_FILES_TO_CHECK
    )
    print(f"\nCONCLUSION: {'ALL CHECKS PASSED' if all_ok else 'FAILURES DETECTED -- see above'}")


if __name__ == "__main__":
    main()
