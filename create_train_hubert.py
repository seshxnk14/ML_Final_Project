import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio import transforms as T


HUBERT_SR = 16000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir", type=str, required=True)
    parser.add_argument("--seglen_ms", type=int, default=200,
                        help="Segment length in ms (default: 200)")
    parser.add_argument("--seghop_ms", type=int, default=100,
                        help="Segment hop in ms (default: 100)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output h5 path; defaults to <traindir>/train_hubert.h5")
    return parser.parse_args()


def normalize(seg: np.ndarray) -> np.ndarray:
    return (seg - seg.mean()) / (seg.std() + 1e-6)


def main():
    cli = parse_args()

    seg_len = int(round(cli.seglen_ms / 1000.0 * HUBERT_SR))
    seg_hop = int(round(cli.seghop_ms / 1000.0 * HUBERT_SR))
    out_path = cli.out or os.path.join(cli.traindir, "train_hubert.h5")
    print(f"seg_len={seg_len} samples ({cli.seglen_ms} ms) "
          f"seg_hop={seg_hop} samples ({cli.seghop_ms} ms)")
    print(f"writing to {out_path}")

    csv_files = sorted(glob.glob(os.path.join(cli.traindir, "*/*.csv")))
    print(f"found {len(csv_files)} CSV files")

    class_names = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for col in df.columns:
            if (
                col not in ("Audiofilename", "Starttime", "Endtime")
                and col not in class_names
                and len(df[df[col] == "POS"]) > 0
            ):
                class_names.append(col)
    cls2int = {c: i for i, c in enumerate(class_names)}
    print(f"discovered {len(class_names)} classes")

    hf = h5py.File(out_path, "w")
    hf.create_dataset("data", shape=(0, seg_len), maxshape=(None, seg_len), dtype="float32")
    hf.create_dataset("label", shape=(0, 1), maxshape=(None, 1), dtype="int64")

    file_index = 0
    for csv_file in csv_files:
        wav_file = csv_file.replace("csv", "wav")
        print(f"processing {wav_file}")

        df = pd.read_csv(csv_file)
        wav, sr = torchaudio.load(wav_file)
        if sr != HUBERT_SR:
            wav = T.Resample(sr, HUBERT_SR)(wav)
        if wav.shape[0] != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.squeeze(0).numpy()

        df_cols = df.columns.tolist()
        dfs = []
        for c in df_cols:
            if len(df[df[c] == "POS"]) > 0:
                dfs.append(df[df[c] == "POS"])
        if not dfs:
            continue
        dfs = pd.concat(dfs)

        for i in range(len(dfs)):
            row = dfs.iloc[i]
            label = None
            for c in df_cols:
                if row[c] == "POS":
                    label = c
                    break
            if label is None:
                continue
            on = max(0, int(round(row["Starttime"] * HUBERT_SR)))
            off = min(len(wav), int(round(row["Endtime"] * HUBERT_SR)))
            if off <= on:
                continue

            start = on
            while off - start >= seg_len:
                seg = wav[start:start + seg_len]
                if not np.allclose(seg, 0):
                    seg = normalize(seg)
                    hf["data"].resize((file_index + 1, seg_len))
                    hf["data"][file_index] = seg.astype(np.float32)
                    hf["label"].resize((file_index + 1, 1))
                    hf["label"][file_index] = cls2int[label]
                    file_index += 1
                start += seg_hop

            tail = wav[start:off]
            if seg_len // 8 < len(tail) < seg_len:
                reps = int(np.ceil(seg_len / max(1, len(tail))))
                tail = np.tile(tail, reps)[:seg_len]
                tail = normalize(tail)
                hf["data"].resize((file_index + 1, seg_len))
                hf["data"][file_index] = tail.astype(np.float32)
                hf["label"].resize((file_index + 1, 1))
                hf["label"][file_index] = cls2int[label]
                file_index += 1

    hf.close()
    print(f"Total segments created: {file_index}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
