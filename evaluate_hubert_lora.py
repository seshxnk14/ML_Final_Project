import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, TensorDataset
from torchaudio import transforms as T
from tqdm import tqdm

from mod1_hubert_lora import DEFAULT_HUBERT, HubertLoRAEncoder

def highpass_filter(wave: torch.Tensor, sr: int = 16000, cutoff: int = 80) -> torch.Tensor:
    return torchaudio.functional.highpass_biquad(wave, sr, float(cutoff))

warnings.filterwarnings(
    "ignore",
    message="At least one mel filterbank has all zero values",
    category=UserWarning,
)

HUBERT_SR = 16000



# Argument parser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valdir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--nshot", type=int, default=5)
    parser.add_argument("--qbs", type=int, default=8)
    parser.add_argument("--ftbs", type=int, default=8)
    parser.add_argument("--ftepochs", type=int, default=20)
    parser.add_argument("--ftlr", type=float, default=3e-4)
    parser.add_argument("--ft", type=int, default=1)
    parser.add_argument("--max_support", type=int, default=20)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attn_epochs", type=int, default=40)
    parser.add_argument("--attn_lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--readout", type=str, default="ce", choices=["ce", "attention", "both"])
    parser.add_argument("--query_hop_mult", type=int, default=2,
                        help="Use a coarser stride for query windows only to reduce FP.")
    parser.add_argument("--min_event_sec", type=float, default=0.15,
                        help="Suppress predicted positive runs shorter than this.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_HUBERT)
    parser.add_argument("--win_min_sec", type=float, default=0.2)
    parser.add_argument("--win_max_sec", type=float, default=2.0)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--prob_smooth_sigma", type=float, default=0.0,
                    help="Gaussian smoothing sigma for CE probabilities (0=off)")
    parser.add_argument("--morph_close", type=int, default=0,
                    help="Maximum gap in frames to close between positive runs (0=off)")
    return parser.parse_args()


# Modules

class CEHead(nn.Module):
    """Lightweight external CE classifier — wraps encoder, never mutates it."""
    def __init__(self, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, pooled):
        return self.fc(pooled)


class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, query, support):
        support = support.expand(query.shape[0], -1, -1)
        attn_output, _ = self.attention(query, support, support)
        return self.classifier(attn_output.squeeze(1))

# Audio utilities

def normalize_seg(seg: torch.Tensor) -> torch.Tensor:
    peak = seg.abs().max()
    if peak > 1e-6:
       return seg / peak
    return seg

def take_segments(wave: torch.Tensor, start: int, end: int,
                  win_len: int, seg_hop: int):
    segs = []
    if end <= start:
        return segs
    if end - start > win_len:
        cur = start
        while end - cur > win_len:
            segs.append(normalize_seg(wave[cur:cur + win_len]))
            cur += seg_hop
    
        if end - cur > win_len // 2:
            tail = wave[cur:end]
            pad = torch.zeros(win_len - tail.shape[-1])
            padded = torch.cat([tail, pad])
            segs.append(normalize_seg(padded))
    else:
        if end - start > win_len // 2:
            tail = wave[start:end]
            pad = torch.zeros(win_len - tail.shape[-1])
            padded = torch.cat([tail, pad])
            segs.append(normalize_seg(padded))
    return segs

def cap_support_balanced(features_pos: torch.Tensor,
                         features_neg: torch.Tensor,
                         max_total: int):
    half = max_total // 2
    n_pos = min(features_pos.shape[0], half)
    n_neg = min(features_neg.shape[0], max_total - n_pos)
    if n_pos < half:
        n_neg = min(features_neg.shape[0], max_total - n_pos)
    if n_neg < (max_total - n_pos):
        n_pos = min(features_pos.shape[0], max_total - n_neg)
    return features_pos[:n_pos], features_neg[:n_neg]


def filter_short_runs(labels: np.ndarray, min_run: int) -> np.ndarray:
    if min_run <= 1:
        return labels
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == 1:
            j = i + 1
            while j < n and out[j] == 1:
                j += 1
            if j - i < min_run:
                out[i:j] = 0
            i = j
        else:
            i += 1
    return out


def morphological_close(labels: np.ndarray, max_gap: int = 2) -> np.ndarray:
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == 1:
            j = i + 1
            while j < n and out[j] == 1:
                j += 1
            gap_start = j
            gap_end = j
            while gap_end < n and out[gap_end] == 0:
                gap_end += 1
            if gap_end < n and gap_end - gap_start <= max_gap:
                out[gap_start:gap_end] = 1
            i = gap_end
        else:
            i += 1
    return out


def disable_hubert_masking(encoder: HubertLoRAEncoder):
    """Disable HuBERT's internal masking — only needed during pretraining."""
    try:
        cfg = encoder.backbone.base_model.config
    except AttributeError:
        cfg = encoder.backbone.config
    cfg.mask_time_prob = 0.0
    cfg.mask_feature_prob = 0.0


def extract_pooled(features: torch.Tensor, encoder: HubertLoRAEncoder,
                   device: str, bs: int) -> torch.Tensor:
    """Extract 768-dim pooled HuBERT hidden states in mini-batches."""
    if features.shape[0] == 0:
        return torch.empty(0, encoder.hidden_size, device=device)
    embs = []
    encoder.eval()
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(features), batch_size=bs):
            pooled, _ = encoder(batch.to(device))
            embs.append(pooled.detach().cpu())
    return torch.cat(embs, dim=0)

# Fine-tuning

def finetune_ce(encoder: HubertLoRAEncoder, ce_head: CEHead,
                dl, args):
    encoder.train()
    ce_head.train()

    loss_fn = nn.CrossEntropyLoss()
    trainable = (
        [p for p in encoder.parameters() if p.requires_grad]
        + list(ce_head.parameters())
    )

    optim = torch.optim.AdamW(trainable, lr=args.ftlr)

    for ep in range(args.ftepochs):
        ep_loss = 0.0
        for x, y in dl:
            x, y = x.to(args.device), y.to(args.device)
            optim.zero_grad()
            pooled, _ = encoder(x)
            logits = ce_head(pooled)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            ep_loss += loss.item()
        ep_loss /= max(1, len(dl))
    return encoder, ce_head


def train_attention(attention_clf: AttentionClassifier,
                    support_embeddings: torch.Tensor,
                    pos_emb: torch.Tensor,
                    neg_emb: torch.Tensor,
                    device: str,
                    attn_epochs: int,
                    attn_lr: float):
    """Train attention classifier on support embeddings."""
    support_labels = torch.cat([
        torch.ones(pos_emb.shape[0], dtype=torch.float32, device=device),
        torch.zeros(neg_emb.shape[0], dtype=torch.float32, device=device),
    ], dim=0)

    support_query = torch.cat([pos_emb, neg_emb], dim=0).unsqueeze(1)

    attn_optim = torch.optim.Adam(attention_clf.parameters(), lr=attn_lr)
    loss_fn = nn.BCEWithLogitsLoss()

    attention_clf.train()
    for _ in range(attn_epochs):
        attn_optim.zero_grad()
        logits = attention_clf(support_query, support_embeddings).squeeze(1)
        loss = loss_fn(logits, support_labels)
        loss.backward()
        attn_optim.step()
    attention_clf.eval()
    return attention_clf

def main():
    args = parse_args()

    out_csv = args.out_csv or os.path.join(
        os.path.dirname(args.ckpt), "..", "outputs", "eval_hubert.csv"
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    filenames = sorted(glob.glob(os.path.join(args.valdir, "*/*.csv")))
    print(f"found {len(filenames)} validation files")

    print("\n=== PHASE 1: CPU preprocessing ===")
    file_data = [] 

    for filename in filenames:
        print(f"  [CPU] {os.path.basename(filename)}")
        audio_name = os.path.basename(filename).replace("csv", "wav")
        wav_file = filename.replace("csv", "wav")

        df = pd.read_csv(filename)
        Q_list = df["Q"].to_numpy()
        index_sup = np.where(Q_list == "POS")[0][:args.nshot]
        if len(index_sup) == 0:
            print("    no POS rows, skipping")
            continue

        durations = [
            float(df["Endtime"].iloc[idx] - df["Starttime"].iloc[idx])
            for idx in index_sup
        ]
        mean_dur = float(np.mean(durations))
        win_sec = max(args.win_min_sec, min(args.win_max_sec, mean_dur))
        win_len = int(round(win_sec * HUBERT_SR))
        seg_hop = max(1, win_len // 2)

        wav, sr = torchaudio.load(wav_file)
        if sr != HUBERT_SR:
            wav = T.Resample(sr, HUBERT_SR)(wav)
        if wav.shape[0] != 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wave = wav.squeeze(0).contiguous()
        wave = highpass_filter(wave, sr=HUBERT_SR, cutoff=80)
        del wav

        st_samples = [int(np.floor(t * HUBERT_SR)) for t in df["Starttime"]]
        et_samples = [int(np.floor(t * HUBERT_SR)) for t in df["Endtime"]]

        features_pos = []
        features_neg = []

        features_neg.extend(
            take_segments(wave, 0, st_samples[index_sup[0]], win_len, seg_hop)
        )
        for i, idx in enumerate(index_sup):
            features_pos.extend(
                take_segments(wave, st_samples[idx], et_samples[idx], win_len, seg_hop)
            )
            if i < len(index_sup) - 1:
                features_neg.extend(
                    take_segments(
                        wave,
                        et_samples[idx],
                        st_samples[index_sup[i + 1]],
                        win_len, seg_hop,
                    )
                )

        if not features_pos:
            print("    no POS segments, skipping")
            del wave
            continue

        features_pos = torch.stack(features_pos).contiguous()
        features_neg = (
            torch.stack(features_neg).contiguous()
            if features_neg else torch.empty(0, win_len)
        )

        features_pos, features_neg = cap_support_balanced(features_pos, features_neg, args.max_support)
        print(f"  capped pos={features_pos.shape[0]} neg={features_neg.shape[0]}")

        seg_hop_q = seg_hop * max(1, args.query_hop_mult)
        cur = et_samples[index_sup[-1]]
        features_q = take_segments(wave, cur, wave.shape[0], win_len, seg_hop_q)
        del wave

        if not features_q:
            print(" — no query segments, skipping")
            continue

        features_q = torch.stack(features_q).contiguous()
        str_time_query = et_samples[index_sup[-1]] / HUBERT_SR
        print(f" query={features_q.shape[0]}")

        file_data.append({
            "audio_name": audio_name,
            "features_pos": features_pos,
            "features_neg": features_neg,
            "features_q": features_q,
            "seg_hop_q": seg_hop_q,
            "str_time_query": str_time_query,
        })

    print(f"\nPhase 1 complete: {len(file_data)} files preprocessed into RAM.")
    print("\n=== PHASE 2: GPU inference ===")

    encoder_base = HubertLoRAEncoder(
        method="scl",
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    encoder_base.load_state_dict(ckpt["encoder"], strict=False)
    disable_hubert_masking(encoder_base)
    base_state = {k: v.clone() for k, v in encoder_base.state_dict().items()}
    del encoder_base
    torch.cuda.empty_cache()
    print("Encoder weights cached. Starting per-file GPU loop.\n")

    name_arr = np.array([])
    onset_arr = np.array([])
    offset_arr = np.array([])

    for item in file_data:
        audio_name = item["audio_name"]
        features_pos = item["features_pos"]
        features_neg = item["features_neg"]
        features_q = item["features_q"]
        seg_hop_q = item["seg_hop_q"]
        str_time_query = item["str_time_query"]

        print(f"\n{'=' * 60}\n[GPU] {audio_name}")

        encoder = HubertLoRAEncoder(
            method="scl",
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        ).to(args.device)
        encoder.load_state_dict(base_state)
        disable_hubert_masking(encoder)

        ce_head = None
        if args.ft >= 1:
            ce_head = CEHead(encoder.hidden_size, num_classes=2).to(args.device)
            data_ft = torch.cat([features_pos, features_neg], dim=0)
            label_ft = torch.cat([
                torch.ones(features_pos.shape[0], dtype=torch.long),
                torch.zeros(features_neg.shape[0], dtype=torch.long),
            ], dim=0)
            dl_ft = DataLoader(TensorDataset(data_ft, label_ft),
                               batch_size=args.ftbs, shuffle=True)
            print(f"  CE fine-tuning for {args.ftepochs} epochs...")
            encoder, ce_head = finetune_ce(encoder, ce_head, dl_ft, args)
            ce_head.eval()
            del data_ft, label_ft, dl_ft
            torch.cuda.empty_cache()


        encoder.eval()
        enc_bs = max(1, min(args.ftbs, args.qbs))
        pos_emb = extract_pooled(features_pos, encoder, args.device, enc_bs).to(args.device)
        neg_emb = extract_pooled(features_neg, encoder, args.device, enc_bs).to(args.device)

        if pos_emb.shape[0] == 0:
            print("  empty pos embeddings, skipping")
            del encoder, ce_head
            torch.cuda.empty_cache()
            continue

        support_embeddings = torch.cat([pos_emb, neg_emb], dim=0).unsqueeze(0)
        print(f"  support_embeddings: {tuple(support_embeddings.shape)}")

        attention_clf = None
        if args.readout in ("attention", "both"):
            attention_clf = AttentionClassifier(
                embed_dim=encoder.hidden_size, num_heads=args.num_heads
            ).to(args.device)
            print(f"  training attention classifier for {args.attn_epochs} epochs...")
            attention_clf = train_attention(
                attention_clf, support_embeddings,
                pos_emb, neg_emb,
                args.device, args.attn_epochs, args.attn_lr,
            )

        del features_pos, features_neg
        torch.cuda.empty_cache()

        loader_q = DataLoader(TensorDataset(features_q), batch_size=args.qbs)
        ce_probs_all = []
        attn_probs_all = []
        with torch.no_grad():
            for (xq,) in tqdm(loader_q, desc="  query"):
                pooled, _ = encoder(xq.to(args.device))

                if args.readout in ("ce", "both"):
                    if ce_head is None:
                        raise RuntimeError("readout includes CE but --ft < 1 gave no CE head.")
                    ce_logits = ce_head(pooled)
                    ce_prob = torch.softmax(ce_logits, dim=-1)[:, 1]
                    ce_probs_all.extend(ce_prob.detach().cpu().tolist())

                if args.readout in ("attention", "both"):
                    z_q = pooled.unsqueeze(1)
                    attn_logits = attention_clf(z_q, support_embeddings).squeeze(1)
                    attn_prob = torch.sigmoid(attn_logits)
                    attn_probs_all.extend(attn_prob.detach().cpu().tolist())

        ce_arr = np.array(ce_probs_all, dtype=np.float32) if ce_probs_all else None
        attn_arr = np.array(attn_probs_all, dtype=np.float32) if attn_probs_all else None
        if args.prob_smooth_sigma > 0:
            if ce_arr is not None:
                ce_arr = gaussian_filter1d(ce_arr, sigma=args.prob_smooth_sigma)
            if attn_arr is not None:
                attn_arr = gaussian_filter1d(attn_arr, sigma=args.prob_smooth_sigma)

        if args.readout == "ce":
            labels_pred = (ce_arr > args.threshold).astype(np.int64).tolist()
        elif args.readout == "attention":
            labels_pred = (attn_arr > args.threshold).astype(np.int64).tolist()
        else: 
            labels_pred = ((ce_arr > args.threshold) & (attn_arr > args.threshold)).astype(np.int64).tolist()

        del features_q, support_embeddings, pos_emb, neg_emb
        del attention_clf, ce_head, encoder
        torch.cuda.empty_cache()

        labs = np.array(labels_pred, dtype=np.int64)
        frame_sec = seg_hop_q / HUBERT_SR
        min_run = max(1, int(np.ceil(args.min_event_sec / max(frame_sec, 1e-8))))
        labs = filter_short_runs(labs, min_run)
        if args.morph_close > 0:
            labs = morphological_close(labs, max_gap=args.morph_close)
        krn = np.array([1, -1])
        changes = np.convolve(krn, labs)
        onset_frames = np.where(changes == 1)[0]
        offset_frames = np.where(changes == -1)[0]

        onset = onset_frames * (seg_hop_q / HUBERT_SR) + str_time_query
        offset = offset_frames * (seg_hop_q / HUBERT_SR) + str_time_query

        if len(onset) != len(offset):
            min_n = min(len(onset), len(offset))
            onset = onset[:min_n]
            offset = offset[:min_n]

        name_arr = np.append(name_arr, np.repeat(audio_name, len(onset)))
        onset_arr = np.append(onset_arr, onset)
        offset_arr = np.append(offset_arr, offset)
        print(f"  detections: {len(onset)}")

    pd.DataFrame({
        "Audiofilename": name_arr,
        "Starttime": onset_arr,
        "Endtime": offset_arr,
    }).to_csv(out_csv, index=False)
    print(f"\nSaved predictions to {out_csv}")


if __name__ == "__main__":
    main()
