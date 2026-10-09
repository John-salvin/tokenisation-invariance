"""COMET-22 scores for the ISO 15919 and noise conditions (GPU).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations
import sys, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB_ROOT / "repo" / "cluster_copy"
LIVE = SB_ROOT / "repo" / "panel-pipeline"
sys.path.insert(0, str(LIVE / "src"))
import metrics_panel as MP

BATCH = int(os.environ.get("BATCH", "64"))
OUT = ROB / "results" / "tables"
INTERIM = ROB / "data" / "interim"

CONDITIONS = [
    ("iso", "target_iso", "reference_iso"),
    ("iso_noise02", "target_noise02", "reference_noise02"),
    ("iso_noise05", "target_noise05", "reference_noise05"),
    ("iso_noise10", "target_noise10", "reference_noise10"),
    ("iso_noise20", "target_noise20", "reference_noise20"),
]


def main():
    df = pd.read_parquet(INTERIM / "iso15919_segments.parquet")
    print(f"segments: {len(df)}", flush=True)

    model = MP.load_comet("comet22", device="cuda")
    print("comet22 loaded", flush=True)

    out = df[["segment_id", "lang", "system", "mqm_score", "mqm_valid",
              "cer_target", "cer_reference"]].copy()

    for name, tcol, rcol in CONDITIONS:
        t0 = time.time()
        samples = [{"src": s, "mt": m, "ref": r} for s, m, r in
                   zip(df["source_en"], df[tcol], df[rcol])]
        res = model.predict(samples, batch_size=BATCH, gpus=1, progress_bar=False)
        out[f"comet22_{name}"] = np.asarray(res.scores, dtype=float)
        print(f"  {name}: {len(samples)} scored in {time.time()-t0:.1f}s "
              f"mean={np.mean(res.scores):.4f}", flush=True)
        out.to_csv(OUT / "iso_comet_scores.csv", index=False)

    print("wrote", OUT / "iso_comet_scores.csv")


if __name__ == "__main__":
    main()
