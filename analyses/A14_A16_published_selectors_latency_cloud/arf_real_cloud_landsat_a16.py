# =============================================================================
# ARF REAL-CLOUD CHECK, LANDSAT 8/9 (protocol amendment A16, part b; read-only)
# =============================================================================
# A8 measured observed cloud for Sentinel-2 only. This script adds Landsat 8 and
# 9, whose Collection 2 Level-2 per-pixel quality band (QA_PIXEL, CFMask) is
# readable without charge through Microsoft Planetary Computer (anonymous SAS
# signing). For every primary-design window (8 dry, 8 wet of A5, 8 added of
# A15), every Landsat 8/9 Level-2 scene over the region in the 48-h window is
# found in the Planetary Computer STAC catalogue (collection landsat-c2-l2), and
# QA_PIXEL is read at reduced resolution and sampled at the same 7 points per H3
# resolution-7 tile as A8 (tile_points of the A8 script, checked by SHA-256).
#   QA_PIXEL bits (USGS LSDS-1619, Landsat 8-9 Collection 2 Level 2 Science
#   Product Guide v6.0, Table 6-2): 0 fill, 1 dilated cloud, 2 cirrus (high
#   confidence), 3 cloud (high confidence), 4 cloud shadow (high confidence),
#   5 snow, 6 clear, 7 water.
#   A point is VALID when bit 0 is not set; CLOUDY when bit 1, 3 or 4 is set
#   (sensitivity: or bit 2); otherwise CLEAR.
#   A tile is OBSERVED by a scene if >= 4 of its 7 points are valid; CLEAR if
#   >= half of its valid points are clear; CLOUDY if >= half are cloudy (the
#   rule of A8).
# Reported as in A8: overpasses (one platform within 10 min), the share of
# cloudy tile observations, the share of overpasses that would raise the
# Fail-Signal (fewer than half of their observed tiles clear), pooled per season
# with each overpass weighted by the tiles it observes. PlanetScope is not
# measured: its usable data masks (UDM2) need a Planet licence.
# Writes ARF_batch/real_cloud_landsat_a16/ (overpasses, windows, summary, table).
# Resumable: finished windows are cached in real_cloud_landsat_a16/cache.
#
# Use (Colab):  !pip install -q rasterio h3==4.5.0 pystac-client planetary-computer
#               !python /content/drive/MyDrive/ARF_code/arf_real_cloud_landsat_a16.py
# =============================================================================
import os
import json
import runpy
import hashlib
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

WINDOW_H = 48.0
DECIMATE = 4                                  # 30 m QA_PIXEL read at 120 m (nearest)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"
PLATFORMS = ("landsat-8", "landsat-9")
CLOUD_BITS = (1, 3, 4)
CIRRUS_BIT = 2
FULL_PASS_PCT = 5.0
INDEXES = (("confirmatory_v153_primary_index.csv", "dry"), ("wetseason_v153_a5_index.csv", "wet"),
           ("dryext_v153_a15_index.csv", "dry_added"))
SHA_A8 = "b703c7d08a8c2006af41b453e946994d6b5ea985045041d75b09bce4da953bba"   # the registered A8 script (arf_real_cloud_a8.py)


def point_classes(qa, with_cirrus=False):
    """valid, cloudy (boolean arrays) from QA_PIXEL values."""
    qa = np.asarray(qa).astype(np.uint32)
    valid = (qa & 1) == 0
    bits = CLOUD_BITS + ((CIRRUS_BIT,) if with_cirrus else ())
    mask = sum(1 << b for b in bits)
    cloudy = valid & ((qa & mask) != 0)
    return valid, cloudy


def classify_qa(qa, with_cirrus=False):
    """Per tile: observed, clear, cloudy from (n_tiles, 7) QA_PIXEL values (the tile rule of A8)."""
    valid, cloudy = point_classes(qa, with_cirrus)
    nv = valid.sum(1); ncl = cloudy.sum(1); nclear = nv - ncl
    observed = nv >= 4
    clear = observed & (nclear * 2 >= nv)
    cld = observed & ~clear & (ncl * 2 >= nv)
    return observed, clear, cld


def sample_qa(href, lon, lat, decimate=DECIMATE):
    """QA_PIXEL value at each point (1 = fill where outside the raster)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform
    from affine import Affine
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            h, w = max(1, src.height // decimate), max(1, src.width // decimate)
            arr = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
            T = src.transform * Affine.scale(src.width / w, src.height / h)
            xs, ys = transform("EPSG:4326", src.crs, lon.ravel().tolist(), lat.ravel().tolist())
    cols, rows = ~T * (np.asarray(xs), np.asarray(ys))
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.ones(rows.size, dtype=np.uint32)
    out[ok] = arr[rows[ok], cols[ok]]
    return out.reshape(lon.shape)


def pc_search(bbox, start, end):
    """Landsat 8/9 Level-2 items from Planetary Computer, assets signed anonymously."""
    import planetary_computer
    import pystac_client
    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    s = cat.search(collections=[COLLECTION], bbox=bbox,
                   datetime=f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
                   query={"platform": {"in": list(PLATFORMS)}})
    return [it.to_dict() for it in s.items()]


def qa_href(it):
    a = it.get("assets", {})
    for k in ("qa_pixel", "QA_PIXEL"):
        if k in a and a[k].get("href"):
            return a[k]["href"]
    return None


def window_analysis(items, tiles, start, lon, lat, sampler=sample_qa, time_to=None):
    scenes, fails = [], []
    for it in sorted(items, key=lambda i: str(i.get("properties", {}).get("datetime", ""))):
        p = it.get("properties", {})
        if str(p.get("platform", "")).lower() not in PLATFORMS:
            continue
        dt = datetime.fromisoformat(str(p["datetime"]).replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
        t_h = (dt - start).total_seconds() / 3600.0
        if not (0.0 <= t_h < WINDOW_H):
            continue
        href = qa_href(it)
        if href is None:
            fails.append({"scene": it.get("id"), "error": "no qa_pixel asset"}); continue
        try:
            qa = sampler(href, lon, lat)
        except Exception as e:                      # unreadable asset: recorded, not hidden
            fails.append({"scene": it.get("id"), "error": f"{type(e).__name__}: {e}"[:200]}); continue
        scenes.append({"platform": str(p.get("platform")).lower(), "time_h": t_h, "id": it.get("id"), "qa": qa,
                       "eo_cloud_cover": p.get("eo:cloud_cover")})
    res = {}
    for label, cirrus in (("main", False), ("with_cirrus", True)):
        groups = []
        for s in sorted(scenes, key=lambda s: (s["platform"], s["time_h"])):
            if groups and groups[-1][0]["platform"] == s["platform"] and s["time_h"] - groups[-1][-1]["time_h"] <= 10 / 60:
                groups[-1].append(s)
            else:
                groups.append([s])
        rows, n_obs, n_cloudy, n_clear, obs_sets, clr_sets, times = [], 0, 0, 0, [], [], []
        for g in groups:
            obs = np.zeros(len(tiles), bool); clr = np.zeros(len(tiles), bool); cld = np.zeros(len(tiles), bool)
            for s in g:
                o, c, k_ = classify_qa(s["qa"], cirrus)
                clr |= c; cld |= k_; obs |= o
            cld &= ~clr
            n_o, n_c, n_k = int(obs.sum()), int(clr.sum()), int(cld.sum())
            n_obs += n_o; n_clear += n_c; n_cloudy += n_k
            t = float(np.median([s["time_h"] for s in g]))
            ec = [s["eo_cloud_cover"] for s in g if s["eo_cloud_cover"] is not None]
            rows.append({"platform": g[0]["platform"], "time_h": t, "scenes": len(g), "tiles_observed": n_o,
                         "tiles_clear": n_c, "tiles_cloudy": n_k,
                         "cloudy_pct_of_observed": 100.0 * n_k / n_o if n_o else None,
                         "fail_signal": bool(n_o and n_c * 2 < n_o),
                         "eo_cloud_cover_mean": float(np.mean(ec)) if ec else None})
            obs_sets.append(set(np.flatnonzero(obs))); clr_sets.append(set(np.flatnonzero(clr))); times.append(t)
        n = len(tiles)
        reach = set().union(*obs_sets) if obs_sets else set()
        res[label] = {"overpasses": len(groups), "scenes": len(scenes), "scenes_unreadable": len(fails),
                      "landsat_observed_pct_of_region": 100.0 * len(reach) / n if n else None,
                      "cloudy_obs_pct": 100.0 * n_cloudy / n_obs if n_obs else None,
                      "clear_obs_pct": 100.0 * n_clear / n_obs if n_obs else None,
                      "fail_signal_overpass_pct": 100.0 * np.mean([r["fail_signal"] for r in rows]) if rows else None,
                      "_overpasses": rows, "_fails": fails}
    return res


def windows_to_run(batch):
    out = []
    for name, season in INDEXES:
        p = os.path.join(batch, name)
        if not os.path.exists(p):
            continue
        idx = pd.read_csv(p, dtype=str).fillna("")
        for _, r in idx[idx.status == "done"].iterrows():
            d = r["out_dir"]
            mp = os.path.join(d, "arf_reproducibility_manifest.json")
            if not os.path.exists(mp):
                continue
            cfg = json.load(open(mp)).get("config_overrides") or {}
            if (cfg.get("active_roi"), cfg.get("fleet"), cfg.get("planet_swath_case", "conservative")) != \
                    ("ckgr_okavango", "earth-resources+planet", "conservative") or cfg.get("exclude_satellites") \
                    or cfg.get("cloud_probability_source", "constant") != "constant":
                continue
            out.append({"season": season, "window": str(cfg["analysis_epoch"])[:10],
                        "start": datetime.fromisoformat(str(cfg["analysis_epoch"])[:19]), "run_dir": d})
    if not out:
        return []
    return pd.DataFrame(out).drop_duplicates("window").sort_values("window").to_dict("records")


def main(argv=None, search_fn=None, sampler=sample_qa):
    ap = argparse.ArgumentParser(description="Real Landsat 8/9 cloud over the primary windows (A16, post hoc).")
    ap.add_argument("--code", default="/content/drive/MyDrive/ARF_code")
    ap.add_argument("--batch-dir", default="/content/drive/MyDrive/ARF_batch")
    ap.add_argument("--sha-a8", default=SHA_A8)
    a, _ = ap.parse_known_args(argv)
    p8 = os.path.join(a.code, "arf_real_cloud_a8.py")
    if a.sha_a8:
        assert hashlib.sha256(open(p8, "rb").read()).hexdigest() == a.sha_a8, "A8 script differs from its release"
    A8 = runpy.run_path(p8, run_name="a8")
    search = search_fn or pc_search
    out = os.path.join(a.batch_dir, "real_cloud_landsat_a16")
    cache_dir = os.path.join(out, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    wins, ovs, fails = [], [], []
    for w in windows_to_run(a.batch_dir):
        cp = os.path.join(cache_dir, f"{w['season']}_{w['window']}.json")
        if os.path.exists(cp):
            c = json.load(open(cp)); wins.append(c["row"]); ovs += c["overpasses"]; fails += c["fails"]
            print(f"{w['window']}: from cache", flush=True); continue
        tiles = list(pd.read_csv(os.path.join(w["run_dir"], "arf_mosaic_tiles.csv"), usecols=["tile"])["tile"].astype(str))
        lon, lat = A8["tile_points"](tiles)
        bbox = [float(lon.min()) - 0.1, float(lat.min()) - 0.1, float(lon.max()) + 0.1, float(lat.max()) + 0.1]
        items = search(bbox, w["start"], w["start"] + timedelta(hours=WINDOW_H))
        r = window_analysis(items, tiles, w["start"], lon, lat, sampler=sampler)
        m = r["main"]
        o_w = [{"season": w["season"], "window": w["window"], **o} for o in m["_overpasses"]]
        f_w = [{"window": w["window"], **f} for f in m["_fails"]]
        row = {"season": w["season"], "window": w["window"], "region_tiles": len(tiles),
               **{k: v for k, v in m.items() if not k.startswith("_")},
               "cloudy_obs_pct_with_cirrus": r["with_cirrus"]["cloudy_obs_pct"]}
        wins.append(row); ovs += o_w; fails += f_w
        json.dump({"row": row, "overpasses": o_w, "fails": f_w}, open(cp, "w"), default=str)
        print(json.dumps({k: row[k] for k in ("season", "window", "overpasses", "scenes", "scenes_unreadable",
                                               "landsat_observed_pct_of_region", "cloudy_obs_pct",
                                               "fail_signal_overpass_pct")}, default=str), flush=True)
    assert wins, "no primary-design window found in the indexes"
    W = pd.DataFrame(wins)
    W.to_csv(os.path.join(out, "a16_landsat_windows.csv"), index=False)
    O = pd.DataFrame(ovs)
    O.to_csv(os.path.join(out, "a16_landsat_overpasses.csv"), index=False)
    pd.DataFrame(fails, columns=["window", "scene", "error"]).to_csv(os.path.join(out, "a16_landsat_unreadable.csv"),
                                                                    index=False)
    seasons = {}
    for s, g in W.groupby("season"):
        seasons[s] = {"windows": int(len(g)), "overpasses": int(g.overpasses.sum()), "scenes": int(g.scenes.sum()),
                      "scenes_unreadable": int(g.scenes_unreadable.sum())}
        if not O.empty and (O.season == s).any():
            h = O[O.season == s].copy()
            h["full_pass"] = h.tiles_observed >= FULL_PASS_PCT / 100.0 * W.region_tiles.iloc[0]
            n = int(h.tiles_observed.sum())
            seasons[s]["pooled"] = {
                "overpasses": int(len(h)), "full_passes": int(h.full_pass.sum()), "tile_observations": n,
                "cloudy_obs_pct": 100.0 * h.tiles_cloudy.sum() / n if n else None,
                "fail_signal_overpasses": int(h.fail_signal.sum()),
                "fail_signal_full_passes": int((h.fail_signal & h.full_pass).sum()),
                "passes_le5pct_cloudy": int((h.cloudy_pct_of_observed <= 5).sum()),
                "passes_ge50pct_cloudy": int((h.cloudy_pct_of_observed >= 50).sum())}
    summ = {"label": "POST HOC (amendment A16, part b)", "sensor": "Landsat 8/9 Collection 2 Level-2 QA_PIXEL",
            "source": f"{STAC} collection {COLLECTION}", "cloud_bits": CLOUD_BITS, "sensitivity_adds_bit": CIRRUS_BIT,
            "points_per_tile": 7, "read_resolution_m": 30 * DECIMATE,
            "not_measured": "PlanetScope (UDM2 needs a Planet licence)", "seasons": seasons}
    json.dump(summ, open(os.path.join(out, "a16_landsat_summary.json"), "w"), indent=1, default=str)
    fmt = lambda x, d=1: "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"
    L = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\renewcommand{\\arraystretch}{1.2}",
         "\\caption{Post hoc (amendment A16): observed cloud in the Landsat 8 and 9 scenes of each window "
         "(Collection 2 Level-2 QA\\_PIXEL; cloud, dilated cloud or cloud shadow; 7 points per H3 cell).}",
         "\\label{tab:a16_landsat}", "\\begin{tabular}{|l|c|c|c|c|}", "\\hline",
         "Window & Overpasses & Observed (\\%) & Cloudy (\\%) & Fail-Signal (\\%) \\\\", "\\hline"]
    for r in W.itertuples():
        L += [f"{r.window} ({r.season.replace('_', ' ')}) & {r.overpasses} & {fmt(r.landsat_observed_pct_of_region)} & "
              f"{fmt(r.cloudy_obs_pct)} & {fmt(r.fail_signal_overpass_pct, 0)} \\\\", "\\hline"]
    L += ["\\end{tabular}", "\\end{table}"]
    open(os.path.join(out, "tab_a16_landsat.tex"), "w").write("\n".join(L) + "\n")
    print(json.dumps(seasons, indent=1, default=str))
    print(f"Files: {out}/")
    return W, summ


if __name__ == "__main__":
    main()
