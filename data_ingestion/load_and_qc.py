

import os, glob, time, warnings
from pathlib import Path
from typing    import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import mne



def load_edf_file(file_path: str) -> Optional[mne.io.BaseRaw]:
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            print(f"[LOAD] {Path(file_path).name}")
            return mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception as e:
            print(f"  ✗ Could not read {file_path}: {e}")
            return None


def extract_fpz_cz(raw: mne.io.BaseRaw) -> Optional[np.ndarray]:
    
    picks = None
    if "EEG Fpz-Cz" in raw.ch_names:
        picks = ["EEG Fpz-Cz"]
    else:
        eegs = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
        if eegs:
            picks = [eegs[0]]

    if picks is None:
        print("  ! No EEG channels found")
        return None

    data = raw.get_data(picks=picks)[0]        # 1‑D np.ndarray

    peak_uv = np.max(np.abs(data))
    if peak_uv < 10:                           # Highly likely still in Volts
        print("  ⚠ Detected tiny values — assuming Volts, scaling ×1e6 to µV")
        data = data * 1e6
    else:
        print("  ✓ EEG already in µV (max ≈ %.2f)" % peak_uv)

    print(f"  Signal preview: min={data.min():.2f} µV, max={data.max():.2f} µV, "
          f"std={data.std():.2f} µV")
    return data


def load_hypnogram(hypno_path: str) -> Optional[np.ndarray]:
    
    try:
        ann = mne.read_annotations(hypno_path, verbose=False)
    except Exception as e:
        print(f"  ✗ Cannot read hypnogram {hypno_path}: {e}")
        return None

    mapping = {
        "Sleep stage W": "W",
        "Sleep stage 1": "N1",
        "Sleep stage 2": "N2",
        "Sleep stage 3": "N3",
        "Sleep stage 4": "N3",
        "Sleep stage R": "REM",
        "Movement time": "M",
        "Sleep stage ?": "?"
    }
    labels: List[str] = [mapping.get(desc, "?") for desc in ann.description]
    return np.array(labels)


def qc_report(eeg: np.ndarray,
              epoch_sec: int = 30,
              fs: int = 100) -> Dict[str, float]:
    
    n_ep = len(eeg) // (epoch_sec * fs)
    epochs = eeg[: n_ep * epoch_sec * fs].reshape(n_ep, -1)

    flat = np.std(epochs, axis=1) < 5.0   # µV
    flat_pct = flat.mean() * 100.0

    return {
        "flatline_percentage": flat_pct,
        "min": float(eeg.min()),
        "max": float(eeg.max())
    }



def process_single(psg_path: str,
                   hypno_path: str,
                   out_dir: Path) -> Optional[Dict]:
    rec_id = Path(psg_path).stem.split("-")[0]
    print(f"\n=== {rec_id} ===")

    raw = load_edf_file(psg_path)
    if raw is None:
        return None

    eeg = extract_fpz_cz(raw)
    if eeg is None:
        return None

    labels = load_hypnogram(hypno_path)
    if labels is None or len(labels) == 0:
        print("  ! Empty hypnogram")
        return None

    qc = qc_report(eeg)
    if qc["flatline_percentage"] > 90:
        print(f"  ! >90 % flatline — skip")
        return None

    ep_samp = 100 * 30
    n_ep = min(len(labels), len(eeg)//ep_samp)
    labels = labels[:n_ep]
    epochs = eeg[: n_ep*ep_samp].reshape(n_ep, ep_samp)

    np.save(out_dir/f"{rec_id}_eeg.npy", epochs.astype(np.float32))
    np.save(out_dir/f"{rec_id}_labels.npy", labels)

    return {"recording": rec_id,
            "epochs": n_ep,
            **qc}



def find_pairs(data_dir: Path) -> List[Tuple[str, str]]:
    
    psgs = sorted(data_dir.glob("*-PSG.edf"))
    pairs = []
    for psg in psgs:
        rid = psg.stem.split("-")[0]      # e.g. SC4001E0
        base = rid[:-1]                   # SC4001E
        hypno = None
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            cand = data_dir / f"{base}{letter}-Hypnogram.edf"
            if cand.exists():
                hypno = cand
                break
        if hypno:
            pairs.append((str(psg), str(hypno)))
        else:
            print(f"  ! No hypnogram for {rid}")
    return pairs


def run_pipeline(project_root: Path,
                 max_files: Optional[int] = None) -> None:
    data_dir = project_root / "sleep-cassette"
    out_dir  = project_root / "processed_data"
    out_dir.mkdir(exist_ok=True, parents=True)

    pairs = find_pairs(data_dir)
    if max_files:
        pairs = pairs[:max_files]

    print(f"\nFound {len(pairs)} PSG‑Hypno pairs")

    summary = []
    for psg, hyp in pairs:
        res = process_single(psg, hyp, out_dir)
        if res:
            summary.append(res)

    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(out_dir / "qc_summary.csv", index=False)
        print(f"\n✓ Saved summary for {len(df)} recordings")
    else:
        print("\n✗ No recordings passed QC")



if __name__ == "__main__":


    PROJECT_ROOT = Path(r"C:\Users\Alexander Speer\Desktop\Columbia Spring 2025\AML\Project2")
    run_pipeline(PROJECT_ROOT, max_files=None)  # set max_files for quick tests
