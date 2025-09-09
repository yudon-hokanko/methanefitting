#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODTRAN CSV → HISUI バンド輝度 → C値 → C-CH4二次式の係数導出
出力:
  - D:\methane\ch4_fit_coeffs.json  （a,b,c）
  - D:\methane\band_selection.csv   （選ばれたCH4バンドの中心波長）
"""

import os, csv, json, math, re
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE_DIR      = Path(r"D:\methane")                  
HISUI_SRF_CSV = Path(r"D:\methane\hisui_srf.csv")   
SAVE_COEFFS   = BASE_DIR / "ch4_fit_coeffs.json"
SAVE_BANDS    = BASE_DIR / "band_selection.csv"
AUTO_SELECT_BANDS = True      
N_BANDS_FOR_C = 6            
REF_WL_ALBEDO = 2139.0     
WL_MIN, WL_MAX = 2100.0, 2400.0

def read_modtran_csv(path: Path):
    waves, rads = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2: 
                continue
            try:
                w = float(row[0])
                L = float(row[1])
            except:
                continue
            waves.append(w); rads.append(L)
    if not waves:
        raise RuntimeError(f"数値データが読めません: {path}")
    waves = np.array(waves, float); rads = np.array(rads, float)
    # check unit
    if waves.max() < 10.0:   # determined as µm
        waves = waves * 1000.0
    # Cut out by wavelength range
    m = (waves >= WL_MIN) & (waves <= WL_MAX)
    return waves[m], rads[m]

def load_srf(csv_path: Path):
    """
    read SRF CSV and return {band_id: {'wave':..., 'resp':..., 'center_nm':...}} for each band
    expect style: band,wave_nm,resp
    """
    bands = defaultdict(lambda: {"wave": [], "resp": []})
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # handling column name variations
        key_band = None
        for k in reader.fieldnames:
            if k.lower() in ("band","band_id","bandid"): key_band = k; break
        key_wave = None
        for k in reader.fieldnames:
            if "wave" in k.lower(): key_wave = k; break
        key_resp = None
        for k in reader.fieldnames:
            if "resp" in k.lower() or "srf" in k.lower(): key_resp = k; break
        if not (key_band and key_wave and key_resp):
            raise ValueError("SRF CSV の列名を判別できません（band, wave*, resp* が必要）")

        for row in reader:
            b = int(row[key_band])
            w = float(row[key_wave])
            r = float(row[key_resp])
            bands[b]["wave"].append(w)
            bands[b]["resp"].append(r)

    # Normalization & wavelength of centre
    out = {}
    for b, d in bands.items():
        w = np.array(d["wave"], float)
        r = np.array(d["resp"], float)
        r = np.maximum(r, 0.0)
        if r.sum() == 0:
            continue
        r = r / r.sum()
        center = float((w * r).sum())
        out[b] = {"wave": w, "resp": r, "center_nm": center}
    if not out:
        raise RuntimeError("SRF が空です")
    return out

def convolve_to_bands(wave_hi, rad_hi, srf_dict):
    band_ids = sorted(srf_dict.keys())
    band_L = []
    centers = []
    for b in band_ids:
        w = srf_dict[b]["wave"]; r = srf_dict[b]["resp"]
        # Interpolate high-resoluion spectra onto the SRF axis
        R = np.interp(w, wave_hi, rad_hi, left=np.nan, right=np.nan)
        val = np.nanmean(R * r)  # Weightened average with area 1
        band_L.append(val); centers.append(srf_dict[b]["center_nm"])
    return np.array(band_ids), np.array(centers), np.array(band_L)

# Extracting parameters ftrom file names
RE_BG   = re.compile(r'bg_A(\d{3})_W(\d{4})', re.I)
RE_TEST = re.compile(r'test_A(\d{3})_W(\d{4})_d(\d{3})ppm_L(\d+)m', re.I)

def main():
    # Read SRF 
    srf = load_srf(HISUI_SRF_CSV)
    band_ids = sorted(srf.keys())
    centers  = np.array([srf[b]["center_nm"] for b in band_ids])
    # Nearest band to 2139nm
    idx_ref = int(np.argmin(np.abs(centers - REF_WL_ALBEDO)))

    # Create background/test band brightness
    bg_dict   = {}   # key: (A,W)      -> band_L (np.array)
    test_dict = {}   # key: (A,W,dppm) -> band_L

    for csvf in BASE_DIR.glob("*.csv"):
        name = csvf.name
        m_bg = RE_BG.search(name)
        m_ts = RE_TEST.search(name)
        try:
            w, L = read_modtran_csv(csvf)
        except Exception as e:
            # Ignore other output of MODTRAN
            continue
        b_ids, cen, bandL = convolve_to_bands(w, L, srf)
        if m_bg:
            A = int(m_bg.group(1))/1000.0
            W = int(m_bg.group(2))/1000.0
            bg_dict[(A,W)] = bandL
        elif m_ts:
            A = int(m_ts.group(1))/1000.0
            W = int(m_ts.group(2))/1000.0
            d = int(m_ts.group(3))/100.0
            test_dict[(A,W,d)] = bandL

    if not bg_dict or not test_dict:
        raise RuntimeError("bg_*.csv / test_*.csv が見つかりません。00_run_modtran.ps1 で生成してください。")

    # Automatically choosing CH4 band（Select bands with high sensitivity to Δppm in regression）
    chosen_idx = None
    if AUTO_SELECT_BANDS:
        # For each band, create residuals = Lb - Lav across all A, W, Δ, and select in descending order of regression coefficients for Δ
        # First subtract the corresponding Lb (same A, W background)
        # Ensure consistency across all keys 
        all_keys = []
        for (A,W,d) in test_dict.keys():
            if (A,W) in bg_dict:
                all_keys.append((A,W,d))
        if not all_keys:
            raise RuntimeError("背景とテストの組み合わせが揃っていません")

        # residuals tables: shape = (nbands, nsamples)
        nb = len(band_ids)
        X = []
        Y = []
        for (A,W,d) in all_keys:
            Lb  = bg_dict[(A,W)]
            Lav = test_dict[(A,W,d)]
            # Estimate A at 2139nm → χ correction to nearest Agrid (same procedure as real data)
            # However, in simulation Agrid=A, so χ ≈ 1
            idx = idx_ref
            # Retrieve L_path and L10 to match W
            Lpath = bg_dict.get((0.0, W), None)
            L10   = bg_dict.get((0.10, W), None)
            if Lpath is None or L10 is None:
                # If no reference is provided, skip χ correction
                chi = 1.0
            else:
                A_est = (Lav[idx] - Lpath[idx]) / (L10[idx] - Lpath[idx]) * 0.10
                Agrid = min(bg_dict.keys(), key=lambda kw: (abs(kw[0]-A)+ (abs(kw[1]-W)*0)))[:1]  # dummy
                chi = (A / max(A_est, 1e-6))
            Lres = Lb - Lav * chi
            X.append(d)
            Y.append(Lres)
        X = np.array(X, float)           # shape (nsamples,)
        Y = np.column_stack(Y).T         # shape (nsamples, nbands)

        # Evaluate the slope |β| of the simple regression for each band
        Xc = X - X.mean()
        denom = (Xc**2).sum() + 1e-12
        betas = (Xc[:,None] * (Y - Y.mean(0))).sum(0) / denom
        order = np.argsort(-np.abs(betas))
        chosen_idx = order[:N_BANDS_FOR_C]
    else:
        # N fibers, ordered by proximity to the center at 2248–2298 nm
        mask = (centers >= 2248) & (centers <= 2298)
        cand = np.where(mask)[0]
        if len(cand) < N_BANDS_FOR_C:
            # Nearest Neighbor Interpolation
            order = np.argsort(np.abs(centers - 2273))
            chosen_idx = order[:N_BANDS_FOR_C]
        else:
            chosen_idx = cand[:N_BANDS_FOR_C]

    chosen_centers = centers[chosen_idx]
    # Save
    with SAVE_BANDS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["band_index","center_nm"])
        for i, c in zip(chosen_idx, chosen_centers):
            w.writerow([int(i), float(c)])

    # Calculation of C-value
    Cs = []
    ch4_1km_ppm = []  # Average concentration in the 1km layer (background 1.8 + Δ, but here we enter background value + Δ)
    for (A,W,d) in sorted(test_dict.keys()):
        Lav = test_dict[(A,W,d)]
        Lb  = bg_dict[(A,W)]
        # 2139nm estimate A → correct χ
        idx = idx_ref
        Lpath = bg_dict.get((0.0, W), None)
        L10   = bg_dict.get((0.10, W), None)
        if Lpath is None or L10 is None:
            chi = 1.0
            A_est = A
        else:
            A_est = (Lav[idx] - Lpath[idx]) / (L10[idx] - Lpath[idx]) * 0.10
            chi = A / max(A_est, 1e-6)

        Lres = Lb - Lav * chi
        C = (Lres[chosen_idx].mean() / max(A, 1e-6)) * 100000.0
        Cs.append(C)
        ch4_1km_ppm.append(1.8 + d)   # Interpreted as the "1km average" obtained by adding delta tothe backgroud value of 1.8ppm

    Cs = np.array(Cs, float)
    y  = np.array(ch4_1km_ppm, float)

    # y = a C^2 + b C + c
    X = np.column_stack([Cs**2, Cs, np.ones_like(Cs)])
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a,b,c = [float(v) for v in coeff]

    with SAVE_COEFFS.open("w", encoding="utf-8") as f:
        json.dump({"a":a,"b":b,"c":c,
                   "ref_band_center_nm": float(centers[idx_ref]),
                   "used_band_centers_nm": [float(v) for v in chosen_centers],
                   "note":"1km層: 背景1.8ppm + Δで解釈"}, f, ensure_ascii=False, indent=2)

    print("係数: a={:.6g}, b={:.6g}, c={:.6g}".format(a,b,c))
    print("参照(アルベド)バンド中心: {:.2f} nm".format(centers[idx_ref]))
    print("Cの平均に使った中心波長[nm]:", ", ".join("{:.1f}".format(v) for v in chosen_centers))
    print("保存:", SAVE_COEFFS, SAVE_BANDS)

if __name__ == "__main__":
    main()
