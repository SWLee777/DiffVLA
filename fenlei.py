
import os
import math
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import yaml
from easydict import EasyDict

import open_clip
from models.Necker import Necker
from models.Adapter import Adapter
from models.MapMaker import MapMaker
from datasets.dataset import ChexpertTestDataset, BusiTestDataset, BrainMRITestDataset
from utils.misc_helper import * 

warnings.filterwarnings('ignore')


@torch.no_grad()
def make_vision_tokens_info(model, model_cfg, layers_out):
    img = torch.ones((1, 3, model_cfg['vision_cfg']['image_size'],
                      model_cfg['vision_cfg']['image_size'])).to(model.device)
    _, tokens = model.encode_image(img, layers_out)
    if len(tokens[0].shape) == 3:
        model.token_size = [int(math.sqrt(t.shape[1] - 1)) for t in tokens]
        model.token_c = [t.shape[-1] for t in tokens]
    else:
        model.token_size = [t.shape[2] for t in tokens]
        model.token_c = [t.shape[1] for t in tokens]
    model.embed_dim = model_cfg['embed_dim']
    print("model token size:", model.token_size, " token dim:", model.token_c)


def _load_state(module, ckpt, prefer_keys, fallback_keys=None, strict=False, name=""):
    if fallback_keys is None:
        fallback_keys = []
    picked = None
    for k in prefer_keys + fallback_keys:
        if k in ckpt and isinstance(ckpt[k], dict) and len(ckpt[k]) > 0:
            picked = k
            break
    if picked is None:
        print(f"⚠️ 未找到{name}权重（{prefer_keys + fallback_keys}），保持初始化。")
        return False
    ret = module.load_state_dict(ckpt[picked], strict=strict)
    if isinstance(ret, tuple):
        missing, unexpected = ret
    else:
        missing, unexpected = ret.missing_keys, ret.unexpected_keys
    if missing or unexpected:
        print(f"⚠️ 加载{name}存在 missing={len(missing)}, unexpected={len(unexpected)}（已忽略）")
    tag = "EMA" if "ema" in picked.lower() else "raw"
    print(f"✅ {name} 加载完成（{tag}）: key='{picked}'")
    return True


def zscore(x):
    mu, sd = float(np.mean(x)), float(np.std(x) + 1e-12)
    return (x - mu) / sd


def best_threshold_youden(y_true, y_score):
    fpr, tpr, th = roc_curve(y_true, y_score)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(th[k]), float(tpr[k]), float(fpr[k])


def threshold_at_tpr(y_true, y_score, target_tpr=0.90):
    fpr, tpr, th = roc_curve(y_true, y_score)
    mask = tpr >= float(target_tpr)
    if not np.any(mask):
        k = int(np.argmax(tpr))
    else:
        k = int(np.argmax(mask))
    return float(th[k]), float(tpr[k]), float(fpr[k])


def image_level_score(amap_2d: torch.Tensor, mode="topq", topq_ratio=0.05):
    """amap_2d: [B,H,W]"""
    B = amap_2d.shape[0]
    flat = amap_2d.view(B, -1)
    if mode == "max":
        return flat.max(dim=1).values
    if mode == "mean":
        return flat.mean(dim=1)
    if mode == "topq":
        k = max(1, int(flat.shape[1] * float(topq_ratio)))
        return torch.topk(flat, k=k, dim=1, largest=True, sorted=False).values.mean(dim=1)
    if mode == "hybrid":
        k = max(1, int(flat.shape[1] * float(topq_ratio)))
        topq = torch.topk(flat, k=k, dim=1, largest=True, sorted=False).values.mean(dim=1)
        mmax = flat.max(dim=1).values
        return 0.5 * (topq + mmax)
    # 默认 topq
    k = max(1, int(flat.shape[1] * float(topq_ratio)))
    return torch.topk(flat, k=k, dim=1, largest=True, sorted=False).values.mean(dim=1)


def build_loader(cfg: EasyDict, preprocess, dataset_name: str):
    root = os.path.join(cfg.data_root, dataset_name)
    if dataset_name == 'chexpert':
        ds = ChexpertTestDataset(cfg, root, preprocess)
    elif dataset_name == 'brainmri':
        ds = BrainMRITestDataset(cfg, root, preprocess)
    elif dataset_name == 'busi':
        ds = BusiTestDataset(cfg, root, preprocess)
    else:
        raise NotImplementedError("dataset must in ['chexpert','busi','brainmri']")
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=2, shuffle=False, pin_memory=True)
    return ds, dl


@torch.no_grad()
def collect_scores_for_ckpt(cfg, model, necker, adapter, prompt_maker, map_maker, loader,
                            strategy: str, topq_ratio: float):
    x_scores, y_means, labels = [], [], []
    for batch in loader:
        images = batch['image'].to(model.device)

        _, image_tokens = model.encode_image(images, out_layers=cfg.layers_out)
        vfeat = necker(image_tokens)
        vfeat = adapter(vfeat)
        pfeat = prompt_maker(vfeat)

        out = map_maker(vfeat, pfeat)
        probs = out[0] if isinstance(out, (tuple, list)) else out  # [B,2,H,W]
        probs = torch.softmax(probs, dim=1)
        amap = probs[:, 1, :, :]  # [B,H,W]

        x = image_level_score(amap, mode=strategy, topq_ratio=topq_ratio)
        y = image_level_score(amap, mode="mean",   topq_ratio=topq_ratio)

        x_scores.append(x.cpu().numpy())
        y_means.append(y.cpu().numpy())
        labels.extend(batch['is_anomaly'].cpu().numpy().tolist())

    x_scores = np.concatenate(x_scores, 0)
    y_means  = np.concatenate(y_means, 0)
    labels   = np.array(labels[:x_scores.shape[0]])
    return x_scores, y_means, labels


def draw_scatter(ax, x, y, labels, title, threshold_mode="youden", target_tpr=0.90, z_axis=True, point_size=10):
    thr_func = best_threshold_youden if threshold_mode == "youden" else \
               (lambda y_true, y_score: threshold_at_tpr(y_true, y_score, target_tpr))
    thr, tpr, fpr = thr_func(labels, x)
    auroc = roc_auc_score(labels, x)

    xs = zscore(x) if z_axis else x
    ys = zscore(y) if z_axis else y

    pred = (x >= thr).astype(np.int32)
    tp = (pred == 1) & (labels == 1)
    tn = (pred == 0) & (labels == 0)
    fp = (pred == 1) & (labels == 0)
    fn = (pred == 0) & (labels == 1)

    ax.scatter(xs[tn], ys[tn], s=point_size, alpha=0.35, c="#4C9BE8", label="TN")
    ax.scatter(xs[tp], ys[tp], s=point_size, alpha=0.35, c="#E67C30", label="TP")
    ax.scatter(xs[fp], ys[fp], s=point_size+8, alpha=0.95, c="#4C9BE8",
               edgecolors="k", linewidths=0.6, label=f"FP ({fp.sum()})")
    ax.scatter(xs[fn], ys[fn], s=point_size+8, alpha=0.95, c="#E67C30",
               edgecolors="k", linewidths=0.6, label=f"FN ({fn.sum()})")

    thr_x = (thr - float(np.mean(x))) / (float(np.std(x) + 1e-12)) if z_axis else thr
    ax.axvline(thr_x, linestyle='--', linewidth=1.0, color='gray')

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("Image score (strategy)" + (" [z]" if z_axis else ""))
    ax.set_ylabel("Mean image score" + (" [z]" if z_axis else ""))
    ax.set_title(f"{title}\nAUROC={auroc:.3f}  thr={thr:.4f}  TPR={tpr:.3f}  FPR={fpr:.3f}")
    ax.legend(fontsize=9, loc="best")


def draw_hist(ax, x, labels, title, threshold_mode="youden", target_tpr=0.90):
    thr_func = best_threshold_youden if threshold_mode == "youden" else \
               (lambda y_true, y_score: threshold_at_tpr(y_true, y_score, target_tpr))
    thr, tpr, fpr = thr_func(labels, x)
    auroc = roc_auc_score(labels, x)

    ax.hist(x[labels == 0], bins=40, alpha=0.65, label="normal")
    ax.hist(x[labels == 1], bins=40, alpha=0.65, label="abnormal")
    ax.axvline(thr, linestyle='--', color='gray', linewidth=1.0, label=f"thr={thr:.4f}")
    ax.set_title(f"{title}\nAUROC={auroc:.3f}  TPR={tpr:.3f}  FPR={fpr:.3f}")
    ax.set_xlabel("Image score (strategy)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="best")


def main(args):
    with open(args.config_path) as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    set_seed(seed=cfg.random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess, model_cfg = open_clip.create_model_and_transforms(
        cfg.model_name, cfg.image_size, device=device
    )
    for p in model.parameters():
        p.requires_grad_(False)
    cfg.model_cfg = model_cfg
    make_vision_tokens_info(model, cfg.model_cfg, cfg.layers_out)

    necker = Necker(clip_model=model).to(model.device)
    adapter = Adapter(clip_model=model, target=cfg.model_cfg['embed_dim']).to(model.device)
    if cfg.prompt_maker == 'coop':
        from models.CoOp import PromptMaker
    else:
        raise NotImplementedError("type of prompt must in ['coop']")
    prompt_maker = PromptMaker(
        prompts=cfg.prompts, clip_model=model,
        n_ctx=cfg.n_learnable_token, CSC=cfg.CSC,
        class_token_position=cfg.class_token_positions,
    ).to(model.device)
    map_maker = MapMaker(
        image_size=cfg.image_size,
        num_layers=len(cfg.layers_out),
        num_classes=2,
        temperature=getattr(cfg, "temperature", 0.10),
        use_refine=getattr(cfg, "use_refine", True),
        refine_groups=getattr(cfg, "refine_groups", 1),
    ).to(model.device)

    ds_name = cfg.get("dataset", None)
    if ds_name is None and hasattr(cfg, "test_datasets") and len(cfg.test_datasets) > 0:
        ds_name = cfg.test_datasets[0]
    assert ds_name is not None, "配置中未找到 dataset / test_datasets"
    ds, loader = build_loader(cfg, preprocess, ds_name)
    print(f"==> Use dataset: {ds_name}, size={len(ds)}")

    strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
    ckpt_paths = [p.strip() for p in args.checkpoint_paths.split(",") if p.strip()]
    print(f"[Compare] strategies={strategies}, topq_ratio={args.topq_ratio}, "
          f"threshold_mode={args.threshold_mode}, target_tpr={args.target_tpr}")
    
    fig_scatter, axes_scatter = plt.subplots(len(ckpt_paths), len(strategies),
                                             figsize=(6 * len(strategies), 5.5 * len(ckpt_paths)))
    if len(ckpt_paths) == 1: axes_scatter = np.array([axes_scatter])
    if len(strategies) == 1: axes_scatter = axes_scatter.reshape(len(ckpt_paths), 1)

    for r, ck in enumerate(ckpt_paths):
        print(f"\n===== 加载 {ck} =====")
        ckpt = torch.load(ck, map_location=map_func)
        _load_state(adapter, ckpt,
                    ["adapter_ema_state_dict", "ema_adapter_state_dict"], ["adapter_state_dict"], name="Adapter")
        if hasattr(prompt_maker, "prompt_learner"):
            _load_state(prompt_maker.prompt_learner, ckpt,
                        ["prompt_ema_state_dict", "ema_prompt_state_dict"], ["prompt_state_dict"], name="Prompt")
        _load_state(necker, ckpt,
                    ["necker_ema_state_dict", "ema_necker_state_dict"], ["necker_state_dict"], name="Necker")
        _load_state(map_maker, ckpt,
                    ["map_maker_ema_state_dict", "ema_map_maker_state_dict"], ["map_maker_state_dict"], name="MapMaker")

        for m in [prompt_maker.prompt_learner, adapter, necker, map_maker]:
            m.eval()

        for c, strat in enumerate(strategies):
            x, y, labels = collect_scores_for_ckpt(cfg, model, necker, adapter, prompt_maker, map_maker, loader,
                                                   strategy=strat, topq_ratio=float(args.topq_ratio))
            title = f"{os.path.basename(ck)} / {strat}"
            draw_scatter(axes_scatter[r, c], x, y, labels, title,
                         threshold_mode=args.threshold_mode, target_tpr=float(args.target_tpr),
                         z_axis=True, point_size=10)

    fig_scatter.tight_layout()
    os.makedirs(os.path.dirname(args.out_scatter) or ".", exist_ok=True)
    fig_scatter.savefig(args.out_scatter, dpi=220)
    print(f"✅ 保存散点图到：{args.out_scatter}")

    fig_hist, axes_hist = plt.subplots(len(ckpt_paths), len(strategies),
                                       figsize=(6 * len(strategies), 4.6 * len(ckpt_paths)))
    if len(ckpt_paths) == 1: axes_hist = np.array([axes_hist])
    if len(strategies) == 1: axes_hist = axes_hist.reshape(len(ckpt_paths), 1)

    for r, ck in enumerate(ckpt_paths):
        ckpt = torch.load(ck, map_location=map_func)
        _load_state(adapter, ckpt,
                    ["adapter_ema_state_dict", "ema_adapter_state_dict"], ["adapter_state_dict"], name="Adapter")
        if hasattr(prompt_maker, "prompt_learner"):
            _load_state(prompt_maker.prompt_learner, ckpt,
                        ["prompt_ema_state_dict", "ema_prompt_state_dict"], ["prompt_state_dict"], name="Prompt")
        _load_state(necker, ckpt,
                    ["necker_ema_state_dict", "ema_necker_state_dict"], ["necker_state_dict"], name="Necker")
        _load_state(map_maker, ckpt,
                    ["map_maker_ema_state_dict", "ema_map_maker_state_dict"], ["map_maker_state_dict"], name="MapMaker")
        for m in [prompt_maker.prompt_learner, adapter, necker, map_maker]:
            m.eval()

        for c, strat in enumerate(strategies):
            x, y, labels = collect_scores_for_ckpt(cfg, model, necker, adapter, prompt_maker, map_maker, loader,
                                                   strategy=strat, topq_ratio=float(args.topq_ratio))
            title = f"{os.path.basename(ck)} / {strat}"
            draw_hist(axes_hist[r, c], x, labels, title,
                      threshold_mode=args.threshold_mode, target_tpr=float(args.target_tpr))

    fig_hist.tight_layout()
    os.makedirs(os.path.dirname(args.out_hist) or ".", exist_ok=True)
    fig_hist.savefig(args.out_hist, dpi=220)
    print(f"✅ 保存直方图到：{args.out_hist}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare pooling strategies with MapMaker image scores")
    parser.add_argument("--config_path", type=str, required=True, help="model configs (same as train/test)")
    parser.add_argument("--checkpoint_paths", type=str, required=True,
                        help="comma-separated ckpt paths, e.g. a.pkl,b.pkl")
    parser.add_argument("--strategies", type=str, default="max,mean,topq,hybrid",
                        help="comma-separated: max,mean,topq,hybrid")
    parser.add_argument("--topq_ratio", type=float, default=0.05, help="q for topq/hybrid (e.g., 0.05/0.1/0.2)")
    parser.add_argument("--threshold_mode", type=str, default="youden", choices=["youden", "tpr"],
                        help="youden=Youden J; tpr=fix TPR≥target_tpr")
    parser.add_argument("--target_tpr", type=float, default=0.90, help="used when threshold_mode=tpr")
    parser.add_argument("--out_scatter", type=str, default="scatter_cmp.png")
    parser.add_argument("--out_hist", type=str, default="hist_cmp.png")
    torch.multiprocessing.set_start_method("spawn")
    args = parser.parse_args()
    main(args)
