"""Consolidate the IndicMT Eval workbooks into one long-format parquet.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as C


class IngestError(RuntimeError):
    """Raised when the workbooks do not match the documented schema."""


def _resolve_sheet(sheet_names: list[str], prefix: str) -> str:
    hits = [s for s in sheet_names if s.strip().startswith(prefix.strip())]
    if len(hits) != 1:
        raise IngestError(f"expected exactly 1 sheet for prefix {prefix!r}, got {hits!r}")
    return hits[0]


def _pick(df: pd.DataFrame, aliases: list[str], *, field: str, lang: str,
          required: bool = True) -> pd.Series:
    for a in aliases:
        if a in df.columns:
            return df[a]
    if required:
        raise IngestError(
            f"[{lang}] no column found for field {field!r}; tried {aliases!r}"
        )
    return pd.Series(np.nan, index=df.index, dtype="float64")


_OPTIONAL_BY_LANG = {"comet_qe": {"romanised": {"hin", "tam"}}}


def _build_condition(df: pd.DataFrame, lang: str, condition: str,
                     aliases: dict[str, list[str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["segment_id"] = [f"{lang}_{i:04d}" for i in range(len(df))]
    out["raw_id"] = df.iloc[:, 0].values
    out["lang"] = lang
    out["condition"] = condition

    for field, alias_list in aliases.items():
        allowed = _OPTIONAL_BY_LANG.get(field, {}).get(condition)
        required = True if allowed is None else (lang in allowed)
        out[field] = _pick(df, alias_list, field=field, lang=lang,
                           required=required).values

    mqm_raw = df[C.MQM_COLUMN]
    mqm = pd.to_numeric(mqm_raw, errors="coerce")
    out["mqm_score"] = mqm.values
    out["mqm_valid"] = mqm.notna().values

    out["system"] = df[C.SYSTEM_COLUMN].values
    out["error_types"] = [
        [t for t in row if isinstance(t, str) and t.strip()]
        for row in df[C.ERROR_TYPE_COLUMNS].itertuples(index=False)
    ]
    out["severity_bucket"] = df[C.ERROR_SEVERITY_COLUMNS[0]].values

    out["domain"] = pd.NA
    return out


def build(xlsx: Path, verbose: bool = True) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx)
    frames, dropped = [], {}

    for prefix, lang in C.SHEET_PREFIX_TO_LANG.items():
        sheet = _resolve_sheet(xl.sheet_names, prefix)
        df = xl.parse(sheet)

        for junk, reason in C.KNOWN_JUNK_COLUMNS.items():
            if junk in df.columns:
                dropped.setdefault(lang, []).append((junk, reason))
                df = df.drop(columns=[junk])

        if len(df) != C.N_PER_LANG:
            raise IngestError(f"[{lang}] expected {C.N_PER_LANG} rows, got {len(df)}")

        frames.append(_build_condition(df, lang, "native", C.NATIVE_ALIASES))
        frames.append(_build_condition(df, lang, "romanised", C.ROMANISED_ALIASES))
        if verbose:
            print(f"  [{lang}] sheet {sheet!r}: {len(df)} rows x2 conditions")

    seg = pd.concat(frames, ignore_index=True)
    seg["lang"] = pd.Categorical(seg["lang"], categories=C.LANG_ORDER, ordered=True)
    seg["condition"] = pd.Categorical(seg["condition"], categories=C.CONDITIONS, ordered=True)
    seg = seg.sort_values(["lang", "condition", "segment_id"]).reset_index(drop=True)

    if verbose and dropped:
        print("  dropped known-junk columns:")
        for lang, items in dropped.items():
            for col, reason in items:
                print(f"    [{lang}] {col!r}: {reason}")
    return seg


def assert_schema(seg: pd.DataFrame) -> list[str]:
    checks: list[str] = []

    def chk(name: str, got, want) -> None:
        ok = got == want
        checks.append(f"{'PASS' if ok else 'FAIL'}  {name:52s} got={got!r} want={want!r}")
        if not ok:
            raise IngestError(f"assertion failed: {name}: got {got!r}, want {want!r}")

    for cond in C.CONDITIONS:
        chk(f"rows[{cond}]", int((seg.condition == cond).sum()), C.N_SEGMENTS_TOTAL)
    chk("rows[total]", len(seg), 2 * C.N_SEGMENTS_TOTAL)

    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            n = int(((seg.lang == lang) & (seg.condition == cond)).sum())
            chk(f"rows[{lang},{cond}]", n, C.N_PER_LANG)

    nat = seg[seg.condition == "native"]
    chk("mqm_valid[native]", int(nat.mqm_valid.sum()), C.N_SEGMENTS_MQM_VALID)
    chk("mqm_invalid[native]", int((~nat.mqm_valid).sum()),
        C.N_SEGMENTS_TOTAL - C.N_SEGMENTS_MQM_VALID)

    piv = seg.pivot_table(index="segment_id", columns="condition",
                          values="mqm_score", observed=True)
    same = bool(np.allclose(piv["native"], piv["romanised"], equal_nan=True))
    chk("mqm identical across conditions", same, True)

    rom = seg[seg.condition == "romanised"]
    have = {l for l in C.LANG_ORDER
            if rom.loc[rom.lang == l, "comet_qe"].notna().any()}
    chk("comet_qe[romanised] languages", sorted(have), ["hin", "tam"])
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description="M0: build segments.parquet")
    ap.add_argument("--xlsx", type=Path, default=C.XLSX_MULTI,
                    help="source workbook (default: the FIXED multi-tokeniser one)")
    ap.add_argument("--out", type=Path, default=C.SEGMENTS_PARQUET)
    args = ap.parse_args()

    print(f"M0 ingest: {args.xlsx}")
    seg = build(args.xlsx)

    print("\nassertions:")
    for line in assert_schema(seg):
        print("  " + line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(args.out, index=False)
    meta = {
        "source_workbook": str(args.xlsx),
        "n_rows": len(seg),
        "n_segments": C.N_SEGMENTS_TOTAL,
        "n_mqm_valid": int(seg[seg.condition == "native"].mqm_valid.sum()),
        "languages": C.LANG_ORDER,
        "conditions": C.CONDITIONS,
        "domain_labels_present": False,
        "domain_note": ("IndicMT Eval workbooks carry no domain column; the study design "
                        "domain invariance cannot run on Indic. WMT24 DEU/SPA does have "
                        "one (see data/latin/)."),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {args.out}  ({len(seg):,} rows)")
    print(f"wrote {args.out.with_suffix('.meta.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
