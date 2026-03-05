#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import pandas as pd
import numpy as np

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def find_gemma(user_path):
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates += ["./tools/gemma", "gemma"]

    for c in candidates:
        # if it's "gemma" rely on PATH
        if c == "gemma":
            try:
                subprocess.run(["gemma", "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                return "gemma"
            except FileNotFoundError:
                continue
        else:
            if os.path.exists(c) and os.access(c, os.X_OK):
                return c

    raise FileNotFoundError(
        "Could not find GEMMA binary. Provide --gemma /path/to/gemma, "
        "or place executable at ./tools/gemma, or ensure `gemma` is in PATH."
    )

def read_fam_ids(fam_path: str) -> pd.DataFrame:
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None)
    if fam.shape[1] < 2:
        raise ValueError(f"Unexpected .fam format: {fam_path}")
    fam = fam.iloc[:, :2].copy()
    fam.columns = ["FID", "IID"]
    return fam

def load_pheno(pheno_path: str) -> pd.DataFrame:
    # Accept:
    # 1) FID IID phenotype (with or without header)
    # 2) single-column matrix already (no IDs)
    df = pd.read_csv(pheno_path, sep=r"\s+|\t", engine="python", header=None)

    if df.shape[1] == 1:
        # already matrix
        return df

    # try to detect header by reading again with header=0
    df_h = pd.read_csv(pheno_path, sep=r"\s+|\t", engine="python", header=0)
    cols_lower = [c.lower() for c in df_h.columns.astype(str)]
    if ("fid" in cols_lower) and ("iid" in cols_lower):
        # pick phenotype column: try common names
        for cand in ["phenotype", "pheno", "trait", "y"]:
            if cand in cols_lower:
                ph_col = df_h.columns[cols_lower.index(cand)]
                return df_h.rename(columns={df_h.columns[cols_lower.index("fid")]: "FID",
                                            df_h.columns[cols_lower.index("iid")]: "IID"})[["FID", "IID", ph_col]] \
                           .rename(columns={ph_col: "PHENO"})
        # else use the 3rd column
        if df_h.shape[1] < 3:
            raise ValueError("Phenotype file has FID/IID but no phenotype column.")
        ph_col = df_h.columns[2]
        return df_h.rename(columns={df_h.columns[cols_lower.index("fid")]: "FID",
                                    df_h.columns[cols_lower.index("iid")]: "IID"})[["FID", "IID", ph_col]] \
                   .rename(columns={ph_col: "PHENO"})

    # no header; assume first two columns are FID IID, third is phenotype
    if df.shape[1] < 3:
        raise ValueError("Phenotype file must be either 1-column matrix or 3-column FID IID PHENO.")
    out = df.iloc[:, :3].copy()
    out.columns = ["FID", "IID", "PHENO"]
    return out

def load_covar(covar_path: str) -> pd.DataFrame:
    # Accept:
    # 1) covariates with FID IID PC1 PC2... (with header)
    # 2) matrix already (no IDs)
    df = pd.read_csv(covar_path, sep=r"\s+|\t", engine="python", header=None)

    if df.shape[1] == 1:
        raise ValueError("Covariate matrix must have >=1 covariate columns (not a single column file).")

    # detect header
    df_h = pd.read_csv(covar_path, sep=r"\s+|\t", engine="python", header=0)
    cols_lower = [c.lower() for c in df_h.columns.astype(str)]
    if ("fid" in cols_lower) and ("iid" in cols_lower):
        # keep all columns after IID
        fid_col = df_h.columns[cols_lower.index("fid")]
        iid_col = df_h.columns[cols_lower.index("iid")]
        cov_cols = [c for c in df_h.columns if c not in [fid_col, iid_col]]
        if len(cov_cols) == 0:
            raise ValueError("Covariate file has FID/IID but no covariate columns.")
        out = df_h.rename(columns={fid_col: "FID", iid_col: "IID"})[["FID", "IID"] + cov_cols].copy()
        return out

    # otherwise treat as matrix already (no IDs)
    # We represent it as DataFrame numeric matrix; caller must align by row count externally.
    return df

def write_pheno_matrix(fam_path: str, pheno_path: str, out_matrix_path: str) -> int:
    fam = read_fam_ids(fam_path)
    ph = load_pheno(pheno_path)

    if ph.shape[1] == 1:
        # already matrix; just validate length
        if len(ph) != len(fam):
            raise ValueError(f"Phenotype matrix rows ({len(ph)}) != fam n ({len(fam)}).")
        ph.to_csv(out_matrix_path, sep="\t", index=False, header=False)
        return len(ph)

    # align by IDs
    merged = fam.merge(ph, on=["FID", "IID"], how="left")
    if merged["PHENO"].isna().any():
        missing = merged[merged["PHENO"].isna()].head(5)
        raise ValueError(f"Missing phenotype for some individuals. Examples:\n{missing}")
    y = merged["PHENO"].astype(float).to_numpy()
    pd.DataFrame(y).to_csv(out_matrix_path, sep="\t", index=False, header=False)
    return len(y)

def write_covar_matrix(fam_path: str, covar_path: str, out_matrix_path: str) -> tuple[int, int]:
    fam = read_fam_ids(fam_path)
    cov = load_covar(covar_path)

    if list(cov.columns[:2]) == ["FID", "IID"]:
        cov_cols = [c for c in cov.columns if c not in ["FID", "IID"]]
        merged = fam.merge(cov, on=["FID", "IID"], how="left")
        if merged[cov_cols].isna().any().any():
            bad = merged[merged[cov_cols].isna().any(axis=1)].head(5)
            raise ValueError(f"Missing covariates for some individuals. Examples:\n{bad}")
        X = merged[cov_cols].astype(float).to_numpy()
    else:
        # already matrix
        if len(cov) != len(fam):
            raise ValueError(f"Covariate matrix rows ({len(cov)}) != fam n ({len(fam)}).")
        X = cov.astype(float).to_numpy()

    pd.DataFrame(X).to_csv(out_matrix_path, sep="\t", index=False, header=False)
    return X.shape[0], X.shape[1]

def run(cmd: list[str]) -> None:
    eprint("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def lambda_gc_from_p(pvals: np.ndarray) -> float:
    p = pd.to_numeric(pd.Series(pvals), errors="coerce").to_numpy()
    p = p[np.isfinite(p)]
    p = p[(p > 0) & (p < 1)]
    if len(p) == 0:
        return float("nan")
    # chi-square(1) inverse survival
    from scipy.stats import chi2
    chisq = chi2.isf(p, 1)
    chisq = chisq[np.isfinite(chisq)]
    return float(np.median(chisq) / 0.456)


def main():
    ap = argparse.ArgumentParser(description="Run GEMMA LMM on chr22 with GRM + optional covariates, then compute λGC + QQ.")
    ap.add_argument("--gemma", default=None, help="Path to GEMMA binary (optional).")
    ap.add_argument("--bfile", required=True, help="PLINK prefix (bed/bim/fam), e.g. data_preprocessed/chr22_qc")
    ap.add_argument("--kinship", required=True, help="Kinship matrix (cXX.txt), e.g. output/grm_chr20_22.cXX.txt")
    ap.add_argument("--pheno", required=True, help="Phenotype file: either 3-col FID IID PHENO or 1-col matrix.")
    ap.add_argument("--covar", default=None, help="Covariate file: either 'FID IID ...' table or matrix aligned with .fam")
    ap.add_argument("--out-prefix", required=True, help="GEMMA output prefix, e.g. gemma_baseline")
    ap.add_argument("--outdir", default="output", help="Where GEMMA writes outputs (default: ./output)")
    ap.add_argument("--lmm", default="4", help="GEMMA LMM mode (default: 4)")
    args = ap.parse_args()

    gemma_bin = find_gemma(args.gemma)

    fam_path = args.bfile + ".fam"
    if not os.path.exists(fam_path):
        raise FileNotFoundError(f"Missing .fam: {fam_path}")

    os.makedirs(args.outdir, exist_ok=True)
    tmpdir = os.path.join(args.outdir, "_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    pheno_mat = os.path.join(tmpdir, f"{args.out_prefix}.pheno.txt")
    n = write_pheno_matrix(fam_path, args.pheno, pheno_mat)
    eprint(f"[ok] wrote pheno matrix: {pheno_mat} (n={n})")

    cmd = [gemma_bin,
           "-bfile", args.bfile,
           "-k", args.kinship,
           "-lmm", str(args.lmm),
           "-p", pheno_mat,
           "-o", args.out_prefix]

    cov_mat = None
    if args.covar:
        cov_mat = os.path.join(tmpdir, f"{args.out_prefix}.covar.txt")
        n2, k = write_covar_matrix(fam_path, args.covar, cov_mat)
        eprint(f"[ok] wrote covar matrix: {cov_mat} (n={n2}, k={k})")
        cmd += ["-c", cov_mat]

    # GEMMA writes into ./output by default, but it respects working directory.
    # We'll run with cwd = project root and rely on default 'output/' folder.
    run(cmd)

    assoc_path = os.path.join("output", f"{args.out_prefix}.assoc.txt")
    if not os.path.exists(assoc_path):
        # sometimes GEMMA might write elsewhere; try args.outdir
        assoc_path2 = os.path.join(args.outdir, f"{args.out_prefix}.assoc.txt")
        if os.path.exists(assoc_path2):
            assoc_path = assoc_path2
        else:
            raise FileNotFoundError(f"Cannot find assoc output: {assoc_path} (or {assoc_path2})")

    df = pd.read_csv(assoc_path, sep=r"\s+")
    if "p_wald" not in df.columns:
        raise ValueError(f"assoc file missing p_wald column: columns={list(df.columns)}")

    lam = lambda_gc_from_p(df["p_wald"].values)
    qq_png = os.path.join(args.outdir, f"{args.out_prefix}.qq.png")
    lam = lambda_gc_from_p(df["p_wald"].values)

    print(f"[done] assoc={assoc_path}")
    print(f"[done] lambda_gc={lam:.6f}")
    

if __name__ == "__main__":
    main()