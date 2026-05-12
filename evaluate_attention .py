import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, TensorDataset
from torchaudio import transforms as T
from tqdm import tqdm

from args import args
from da import Compander, RandomCrop, Resize
from models import ResNet
from util import finetune_ce

import warnings

warnings.filterwarnings(
    "ignore",
    message="At least one mel filterbank has all zero values",
    category=UserWarning,
)

MAX_SUPPORT = 20
ATTN_EPOCHS = 40
ATTN_LR = 1e-3


class AttentionClassifier(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, query, support):
        support = support.expand(query.shape[0], -1, -1)
        attn_output, _ = self.attention(query, support, support)
        logits = self.classifier(attn_output.squeeze(1))
        return logits


def encode_in_batches(features, encoder, device, bs):
    if features.shape[0] == 0:
        return torch.empty(0, device=device)
    embs = []
    ds = TensorDataset(features)
    dl = DataLoader(ds, batch_size=bs)
    with torch.no_grad():
        for (batch,) in dl:
            x, _ = encoder(batch.to(device))
            embs.append(x.detach())
    return torch.cat(embs, dim=0)


def cap_support_balanced(features_pos, features_neg, max_total):
    n_pos_total = features_pos.shape[0]
    n_neg_total = features_neg.shape[0]
    half = max_total // 2

    n_pos = min(n_pos_total, half)
    n_neg = min(n_neg_total, max_total - n_pos)

    if n_pos < half:
        n_neg = min(n_neg_total, max_total - n_pos)
    if n_neg < (max_total - n_pos):
        n_pos = min(n_pos_total, max_total - n_neg)

    return features_pos[:n_pos], features_neg[:n_neg]


def filter_short_runs(labels, min_run):
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


if args.wandb:
    import wandb

torch.cuda.empty_cache()

val_dir = args.valdir
out_dir = os.path.join(args.traindir, '../../outputs')
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, 'eval.csv')
ckpt_dir = os.path.join(args.traindir, '../model/')
TARGET_SR = args.sr
N_MELS = args.nmels
HOP_LEN = args.hoplen
N_SHOT = args.nshot
BATCH_SIZE = args.qbs
NUM_WORKERS = args.workers
fps = TARGET_SR / HOP_LEN

mel = T.MelSpectrogram(
    sample_rate=args.sr,
    n_fft=args.nfft,
    hop_length=args.hoplen,
    f_min=args.fmin,
    f_max=args.fmax,
    n_mels=args.nmels,
)
power_to_db = T.AmplitudeToDB()
transform = nn.Sequential(mel, power_to_db)

name_arr = np.array([])
onset_arr = np.array([])
offset_arr = np.array([])
filenames = [file for file in glob.glob(os.path.join(val_dir, '*/*.csv'))]

for filename in filenames:
    print(f"\n{'=' * 60}")
    print(f"processing file {filename}")
    feat_name = filename.split('/')[-1]
    audio_name = feat_name.replace('csv', 'wav')
    csv_file = os.path.join(val_dir, filename)
    wav_file = csv_file.replace('csv', 'wav')

    df = pd.read_csv(csv_file)
    df.loc[:, 'Starttime'] = df['Starttime']
    df.loc[:, 'Endtime'] = df['Endtime']
    Q_list = df['Q'].to_numpy()
    index_sup = np.where(Q_list == 'POS')[0][:N_SHOT]
    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]
    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]

    difference = []
    for index in index_sup:
        difference.append(end_time[index] - start_time[index])
    max_len = int(round(np.mean(difference)))

    if max_len <= 17:
        win_len = 17
    elif max_len <= 100:
        win_len = max_len
    elif max_len <= 200:
        win_len = max_len // 2
    elif max_len <= 400:
        win_len = max_len // 4
    else:
        win_len = max_len // 8
    seg_hop = win_len // 2

    print(f"win_len={win_len}, seg_hop={seg_hop}")
    print("Loading and resampling audio")

    wav, sr = torchaudio.load(wav_file)
    resample = T.Resample(sr, TARGET_SR)
    wav = resample(wav)
    melspec = transform(wav)
    print(f"melspec shape: {melspec.shape}")

    features_pos = []
    features_neg = []

    end_t0 = start_time[index_sup[0]]
    curr_t0 = 0
    if end_t0 - curr_t0 > win_len:
        while end_t0 - curr_t0 > win_len:
            spec = melspec[..., curr_t0:curr_t0 + win_len]
            curr_t0 += seg_hop
            features_neg.append(spec)
        if end_t0 - curr_t0 > win_len // 2:
            spec = melspec[..., curr_t0:curr_t0 + win_len]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1, 1, repeat_num)
            spec = spec[..., :int(win_len)]
            features_neg.append(spec)
    else:
        if end_t0 - curr_t0 > 0:
            spec = melspec[..., curr_t0:curr_t0 + win_len]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1, 1, repeat_num)
            spec = spec[..., :int(win_len)]
            features_neg.append(spec)

    for index in range(len(index_sup)):
        start_idx = start_time[index_sup[index]]
        if end_time[index_sup[index]] - start_idx > win_len:
            while end_time[index_sup[index]] - start_idx > win_len:
                spec = melspec[..., start_idx:start_idx + win_len]
                start_idx += seg_hop
                features_pos.append(spec)
            if end_time[index_sup[index]] - start_idx > win_len // 2:
                spec = melspec[..., start_idx:end_time[index_sup[index]]]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1, 1, repeat_num)
                spec = spec[..., :int(win_len)]
                features_pos.append(spec)
        else:
            if end_time[index_sup[index]] - start_idx > 0:
                spec = melspec[..., start_idx:end_time[index_sup[index]]]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1, 1, repeat_num)
                spec = spec[..., :int(win_len)]
                features_pos.append(spec)


        start_idx = end_time[index_sup[index]]
        if index < len(index_sup) - 1:
            if start_time[index_sup[index + 1]] - start_idx > win_len:
                while start_time[index_sup[index + 1]] - start_idx > win_len:
                    spec = melspec[..., start_idx:start_idx + win_len]
                    start_idx += seg_hop
                    features_neg.append(spec)
                if start_time[index_sup[index + 1]] - start_idx > win_len // 2:
                    spec = melspec[..., start_idx:start_time[index_sup[index + 1]]]
                    repeat_num = int(win_len / (spec.shape[-1])) + 1
                    spec = spec.repeat(1, 1, repeat_num)
                    spec = spec[..., :int(win_len)]
                    features_neg.append(spec)
            else:
                if start_time[index_sup[index + 1]] - start_idx > 0:
                    spec = melspec[..., start_idx:start_time[index_sup[index + 1]]]
                    repeat_num = int(win_len / (spec.shape[-1])) + 1
                    spec = spec.repeat(1, 1, repeat_num)
                    spec = spec[..., :int(win_len)]
                    features_neg.append(spec)

    print(f"raw pos segments: {len(features_pos)}, raw neg segments: {len(features_neg)}")
    features_pos = torch.stack(features_pos)
    features_neg = torch.stack(features_neg)

    features_pos, features_neg = cap_support_balanced(features_pos, features_neg, MAX_SUPPORT)
    print(
        f"capped to: pos={features_pos.shape[0]}, neg={features_neg.shape[0]} "
        f"(MAX_SUPPORT={MAX_SUPPORT})"
    )

    features_q = []
    last_frame = melspec.shape[-1]
    curr_frame = end_time[index_sup[-1]]
    print(f"Building query segments (last_frame={last_frame}, curr_frame={curr_frame})...")

    if last_frame - curr_frame > win_len:
        while last_frame - curr_frame > win_len:
            spec = melspec[..., curr_frame:curr_frame + win_len]
            features_q.append(spec)
            curr_frame += seg_hop
        if last_frame - curr_frame > win_len // 2:
            spec = melspec[..., curr_frame:last_frame]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1, 1, repeat_num)
            spec = spec[..., :int(win_len)]
            features_q.append(spec)
    else:
        if last_frame - curr_frame > win_len // 2:
            spec = melspec[..., curr_frame:last_frame]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1, 1, repeat_num)
            spec = spec[..., :int(win_len)]
            features_q.append(spec)

    print(f"query segments before stack: {len(features_q)}")
    features_q = torch.stack(features_q)
    print(f"features_q shape: {features_q.shape}")

    del melspec, wav
    torch.cuda.empty_cache()

    rc = RandomCrop(n_mels=128, time_steps=features_q.shape[-1], tcrop_ratio=0.9)
    resize = Resize(n_mels=128, time_steps=features_q.shape[-1])
    comp = Compander(comp_alpha=0.9) 
    if args.ft == 0:
        makeview = nn.Identity()
    else:
        makeview = nn.Sequential(rc, resize)

    print("Loading encoder and fine-tuning")
    encoder = ResNet(method='ce', num_classes=2)
    ckpt = torch.load(os.path.join(ckpt_dir, 'ckpt.pth'), map_location=torch.device('cpu'))
    encoder.load_state_dict(ckpt['encoder'], strict=False)
    encoder = encoder.to(args.device)

    data_finetune = torch.cat([features_pos, features_neg], dim=0)
    label_finetune = torch.cat(
        [
            torch.ones(features_pos.shape[0], dtype=torch.long),
            torch.zeros(features_neg.shape[0], dtype=torch.long),
        ],
        dim=0,
    )

    bs_finetune = args.ftbs
    ds_finetune = TensorDataset(data_finetune, label_finetune)
    dl_finetune = DataLoader(ds_finetune, batch_size=bs_finetune, shuffle=True)
    encoder = finetune_ce(encoder, dl_finetune, makeview, args)

    print("Extracting support embeddings")
    encoder.eval()
    enc_bs = max(1, min(8, BATCH_SIZE))
    pos_emb = encode_in_batches(features_pos, encoder, args.device, enc_bs)
    neg_emb = encode_in_batches(features_neg, encoder, args.device, enc_bs)
    support_embeddings = torch.cat([pos_emb, neg_emb], dim=0).unsqueeze(0)
    print(f"support_embeddings shape: {tuple(support_embeddings.shape)}")

    support_labels = torch.cat(
        [
            torch.ones(pos_emb.shape[0], dtype=torch.float32, device=args.device),
            torch.zeros(neg_emb.shape[0], dtype=torch.float32, device=args.device),
        ],
        dim=0,
    )

    del features_pos, features_neg, data_finetune, label_finetune, ds_finetune, dl_finetune
    torch.cuda.empty_cache()

    embed_dim = support_embeddings.shape[-1]
    attention_clf = AttentionClassifier(embed_dim=embed_dim, num_heads=4).to(args.device)
    attention_clf.train()

    attn_optim = torch.optim.Adam(attention_clf.parameters(), lr=ATTN_LR)
    attn_loss_fn = nn.BCEWithLogitsLoss()
    support_query = torch.cat([pos_emb, neg_emb], dim=0).unsqueeze(1)
    for _ in range(ATTN_EPOCHS):
        attn_optim.zero_grad()
        logits_sup = attention_clf(support_query, support_embeddings).squeeze(1)
        loss_sup = attn_loss_fn(logits_sup, support_labels)
        loss_sup.backward()
        attn_optim.step()
    attention_clf.eval()

    print(f"Running attention inference on {len(features_q)} query segments")
    ds_q = TensorDataset(features_q)
    loader_q = DataLoader(ds_q, batch_size=BATCH_SIZE)

    labels_pred = []
    with torch.no_grad():
        for x_q in tqdm(loader_q):
            x_q_emb, ce_out = encoder(x_q[0].to(args.device))
            x_q_emb = x_q_emb.unsqueeze(1)
            attn_logits = attention_clf(x_q_emb, support_embeddings).squeeze(1)
            attn_pred = torch.sigmoid(attn_logits) > args.decision_threshold
            ce_pred = torch.argmax(ce_out, dim=-1) == 1
            if args.readout == "ce":
                label_pred = ce_pred.long().cpu()
            elif args.readout == "attention":
                label_pred = attn_pred.long().cpu()
            else:
                label_pred = (ce_pred & attn_pred).long().cpu()
            labels_pred.extend(label_pred.tolist())

    del features_q, pos_emb, neg_emb, support_embeddings
    del attention_clf, encoder
    torch.cuda.empty_cache()

    labs_pred = np.array(labels_pred, dtype=np.int64)
    frame_sec = seg_hop * HOP_LEN / TARGET_SR
    min_run = max(1, int(np.ceil(args.min_event_sec / max(frame_sec, 1e-8))))
    labs_pred = filter_short_runs(labs_pred, min_run)

    krn = np.array([1, -1])
    changes = np.convolve(krn, labs_pred)
    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = end_time[index_sup[-1]] * HOP_LEN / TARGET_SR
    onset = onset_frames * seg_hop * HOP_LEN / TARGET_SR + str_time_query
    offset = offset_frames * seg_hop * HOP_LEN / TARGET_SR + str_time_query

    assert len(onset) == len(offset)

    name = np.repeat(audio_name, len(onset))
    name_arr = np.append(name_arr, name)
    onset_arr = np.append(onset_arr, onset)
    offset_arr = np.append(offset_arr, offset)
    print(f"Done. Detections: {len(onset)}")

df_out = pd.DataFrame({
    'Audiofilename': name_arr,
    'Starttime': onset_arr,
    'Endtime': offset_arr,
})
df_out.to_csv(csv_path, index=False)
print(f"\nOutput saved to {csv_path}")
