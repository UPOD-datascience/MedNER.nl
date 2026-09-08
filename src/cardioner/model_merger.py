"""
 Model merger

 Given a list of folder with transformer models

 Accumulate the model weights and average


[Example](https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008)
```
Example:

modelA = nn.Linear(1, 1)
modelB = nn.Linear(1, 1)

sdA = modelA.state_dict()
sdB = modelB.state_dict()

# Average all parameters
for key in sdA:
    sdB[key] = (sdB[key] + sdA[key]) / 2.

# Recreate model and load averaged state_dict (or use modelA/B)
model = nn.Linear(1, 1)
model.load_state_dict(sdB)
```

Input:
- main_model_folder with
 - model.bin's
 - model folders
- output_dir

Output:
- averaged_model
"""

import argparse
import gc
import glob
import json
import math
import os
import shutil
from collections import defaultdict
from os import environ
from typing import Dict, List, Literal, Optional, Set

import torch
import transformers


SUPPORTED_FORCED_MODEL_TYPES = {
    "bert": (
        transformers.BertConfig,
        transformers.BertForTokenClassification,
        "BertForTokenClassification",
    ),
    "roberta": (
        transformers.RobertaConfig,
        transformers.RobertaForTokenClassification,
        "RobertaForTokenClassification",
    ),
}


def normalize_model_type(model_type: Optional[str]) -> Optional[str]:
    if model_type is None:
        return None
    mt = model_type.strip().lower().replace("_", "-")
    aliases = {
        "robert": "roberta",  # common typo
    }
    mt = aliases.get(mt, mt)
    if mt not in SUPPORTED_FORCED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported forced model type '{model_type}'. "
            f"Choose one of: {sorted(SUPPORTED_FORCED_MODEL_TYPES.keys())}"
        )
    return mt


def load_model_config(model_dir: str, force_model_type: Optional[str] = None):
    force_model_type = normalize_model_type(force_model_type)
    if force_model_type is None:
        return transformers.AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    # Read raw config dict so we can recover from malformed/unknown model_type tags.
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Could not find config.json in '{model_dir}' for forced model type loading."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    config_cls, _, arch_name = SUPPORTED_FORCED_MODEL_TYPES[force_model_type]
    cfg_dict["model_type"] = force_model_type
    cfg_dict["architectures"] = [arch_name]
    return config_cls.from_dict(cfg_dict)


def load_token_cls_model(model_dir: str, force_model_type: Optional[str] = None):
    force_model_type = normalize_model_type(force_model_type)
    if force_model_type is None:
        return transformers.AutoModelForTokenClassification.from_pretrained(
            model_dir, trust_remote_code=True
        )

    config = load_model_config(model_dir=model_dir, force_model_type=force_model_type)
    _, model_cls, _ = SUPPORTED_FORCED_MODEL_TYPES[force_model_type]
    return model_cls.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
    )


def load_state_dict(model_dir: str, force_model_type: Optional[str] = None):
    # Load a full model (handles shards, safetensors/bin) then grab its sd.
    model = load_token_cls_model(
        model_dir=model_dir, force_model_type=force_model_type
    )
    sd = model.state_dict()
    del model
    gc.collect()
    return sd


def average_state_dict_advanced(
    model_dirs: List[str],
    method: str = "chordal",  # "chordal" | "karcher" | "chain"
    weights: Optional[
        List[float]
    ] = None,  # same length as model_dirs (chordal/karcher)
    chain_ts: Optional[List[float]] = None,  # length len(model_dirs)-1 (chain)
    keep_missing: bool = False,  # carry through tensors not shared by all
    force_model_type: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load checkpoints from each directory in `model_dirs`, average them on a per-tensor basis,
    and return a *plain* state_dict (no wrapper). Supports:
      - PyTorch files: *.pt, *.pth, *.bin (prefers 'pytorch_model.bin')
      - SafeTensors: *.safetensors (if 'safetensors' is installed)

    Methods:
      - chordal : fast spherical barycenter approximation (default)
      - karcher : iterative Riemannian mean (slower, more exact)
      - chain   : pairwise SLERP over the list using `chain_ts`

    Notes:
      - Only float tensors with matching shapes are averaged; others fall back to a sensible pick.
      - Computation is in double for stability, cast back to original dtype.
    """

    # check if gpu available and assign device
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------- helpers -----------------
    try:
        from safetensors.torch import load_file as load_safetensors

        _has_safetensors = True
    except Exception:
        _has_safetensors = False

    def _check_model_folder(d: str) -> bool:
        # Priority order typical for HF and PyTorch
        priority = [
            "*.safetensors",
            "pytorch_model.bin",
            "model.bin",
            "*.pt",
            "*.pth",
            "*.bin",
        ]
        for pat in priority:
            paths = []
            if "*" in pat:
                paths = glob.glob(os.path.join(d, pat))
            else:
                p = os.path.join(d, pat)
                if os.path.isfile(p):
                    paths = [p]
            # Prefer exact matches, then first match
            if paths:
                # If both .bin and .safetensors exist, we'll still respect priority order
                return True
        return False

    # def _load_state_from_path(path: str):
    #     ext = os.path.splitext(path)[1].lower()
    #     if ext == ".safetensors":
    #         if not _has_safetensors:
    #             raise RuntimeError("Found .safetensors but the 'safetensors' package is not installed.")
    #         sd = load_safetensors(path, device="cpu")
    #     else:
    #         sd = torch.load(path, map_location="cpu")
    #     # Unwrap {"state_dict": ...} if necessary
    #     if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
    #         sd = sd["state_dict"]
    #     if not isinstance(sd, dict):
    #         raise ValueError(f"Unsupported checkpoint structure at {path}")
    #     return sd

    def _angle(u, v):
        un, vn = u.norm(), v.norm()
        if un < 1e-12 or vn < 1e-12:
            return 0.0
        cosw = torch.clamp((u @ v) / (un * vn), -1.0, 1.0)
        return float(math.acos(cosw))

    def _normalize(x):
        n = x.norm()
        return x if n < 1e-12 else x / n

    def _slerp_vec(a, b, t):
        an, bn = a.norm(), b.norm()
        if an < 1e-12 or bn < 1e-12:
            return (1 - t) * a + t * b
        w = _angle(a, b)
        if w < 1e-7:
            return (1 - t) * a + t * b
        so = math.sin(w)
        return (math.sin((1 - t) * w) / so) * a + (math.sin(t * w) / so) * b

    def _log_map(p, q):
        p = _normalize(p)
        q = _normalize(q)
        w = _angle(p, q)
        if w < 1e-7:
            return torch.zeros_like(p)
        v = q - (p @ q) * p
        nv = v.norm()
        if nv < 1e-12:
            return torch.zeros_like(p)
        return v * (w / nv)

    def _exp_map(p, v):
        pv = v.norm()
        if pv < 1e-12:
            return p
        p = _normalize(p)
        return math.cos(pv) * p + math.sin(pv) * (v / pv)

    def _to_vec(x):
        return x.reshape(-1).double()

    def _from_vec(x, shape, dtype):
        return x.reshape(shape).to(dtype)

    def _slerp_tensor(A, B, t):
        v = _slerp_vec(_to_vec(A), _to_vec(B), t)
        return _from_vec(v, A.shape, A.dtype)

    def _safe_unit(x: torch.Tensor) -> torch.Tensor:
        v = _to_vec(x)
        n = v.norm()
        return v if n < 1e-12 else v / n, float(n)

    def _chordal_barycenter(tensors: List[torch.Tensor], wts: List[float]):
        # direction (unit) + radius preservation
        units, norms = zip(*[_safe_unit(t) for t in tensors])
        w = torch.tensor(wts, dtype=torch.double)
        w = w / w.sum()

        # weighted Euclidean mean of unit vectors, then renormalize => fast spherical approx
        m = torch.stack(units, 0).mul(w.view(-1, 1)).sum(0)
        if m.norm() < 1e-12:
            # degenerate: fall back to weighted Euclidean avg of original tensors
            return sum(t * wi for t, wi in zip(tensors, w.tolist()))
        u_mean = m / m.norm()

        # **preserve scale**: weighted arithmetic radius (works well in practice)
        r = float(torch.tensor(norms, dtype=torch.double).dot(w))
        return _from_vec(r * u_mean, tensors[0].shape, tensors[0].dtype)

    def _karcher_mean(
        tensors: List[torch.Tensor], wts: List[float], iters=10, tol=1e-7
    ):
        # Karcher mean **on unit vectors**, then re-apply radius
        units, norms = zip(*[_safe_unit(t) for t in tensors])
        w = torch.tensor(wts, dtype=torch.double)
        w = w / w.sum()

        # init with chordal direction
        m = torch.stack(units, 0).mul(w.view(-1, 1)).sum(0)
        if m.norm() < 1e-12:
            # if all near-zero, fallback to weighted Euclidean avg of originals
            return sum(t * wi for t, wi in zip(tensors, w.tolist()))
        m = m / m.norm()

        for _ in range(iters):
            logs = torch.stack(
                [_log_map(m, q) * wi for q, wi in zip(units, w.tolist())], 0
            ).sum(0)
            if logs.norm() < tol:
                break
            m = _exp_map(m, logs)
            # keep on sphere
            if m.norm() > 0:
                m = m / m.norm()

        # **preserve scale**: arithmetic radius (or try geometric if you like)
        r = float(torch.tensor(norms, dtype=torch.double).dot(w))
        return _from_vec(r * m, tensors[0].shape, tensors[0].dtype)

    def _use_spherical(key: str, t: torch.Tensor) -> bool:
        k = key.lower()
        if any(
            s in k
            for s in [
                "bias",
                "layernorm.weight",
                "layernorm.bias",
                "ln_",
                ".norm.",
                "norm.weight",
                "norm.bias",
            ]
        ):
            return False
        # tiny or 1D params are often scales/offsets; lerp them
        if t.ndim <= 1 or t.numel() < 32:
            return False
        return True

    # ----------------- load checkpoints -----------------
    if len(model_dirs) < 2:
        raise ValueError("Provide at least two model directories.")

    ckpts = []
    for d in model_dirs:
        path = _check_model_folder(d)
        if not path:
            raise FileNotFoundError(f"No checkpoint file found in: {d}")
        ckpts.append(load_state_dict(d, force_model_type=force_model_type))

    # ----------------- prepare keys & weights -----------------
    all_keys = set().union(*[set(s.keys()) for s in ckpts])
    inter_keys = set(ckpts[0].keys())
    for s in ckpts[1:]:
        inter_keys &= set(s.keys())

    if method not in {"chordal", "karcher", "chain"}:
        raise ValueError("method must be one of: chordal, karcher, chain")

    if method == "chain":
        n = len(ckpts)
        if chain_ts is not None:
            if len(chain_ts) != n - 1:
                raise ValueError("chain_ts must have length len(model_dirs)-1")
            ts = list(map(float, chain_ts))
        else:
            #ts = [1.0 / (n - i + 1) for i in range(1, n)]  # gentle progression
            ts = [1.0 / (i + 1) for i in range(1, n)]
    else:
        if weights is None:
            weights = [1.0 / len(ckpts)] * len(ckpts)
        if len(weights) != len(ckpts):
            raise ValueError("weights length must equal number of model dirs")
        weights = list(map(float, weights))

    # ----------------- compute averaged state_dict -----------------
    out: dict[str, torch.Tensor] = {}

    # Shared tensors → averaged
    for k in sorted(inter_keys):
        vals = [
            s[k].to(device) if isinstance(s[k], torch.Tensor) else s[k] for s in ckpts
        ]
        if all(
            isinstance(v, torch.Tensor) and v.dtype.is_floating_point for v in vals
        ) and all(v.shape == vals[0].shape for v in vals):
            if method == "chain":
                # keep your existing SLERP (already preserves radius)
                acc = vals[0]
                for i in range(1, len(vals)):
                    # optionally: force LERP for fragile keys even in chain
                    acc = (
                        _slerp_tensor(acc, vals[i], ts[i - 1])
                        if _use_spherical(k, vals[0])
                        else ((1 - ts[i - 1]) * acc + ts[i - 1] * vals[i])
                    )
                out[k] = acc
            elif _use_spherical(k, vals[0]):
                out[k] = (
                    _chordal_barycenter(vals, weights)
                    if method == "chordal"
                    else _karcher_mean(vals, weights)
                )
            else:
                # arithmetic average for fragile params
                w = torch.tensor(weights, dtype=vals[0].dtype, device=vals[0].device)
                w = w / w.sum()
                out[k] = sum(v * wi for v, wi in zip(vals, w.tolist()))
        else:
            out[k] = vals[0]

    # Optional: carry tensors that aren’t shared by all
    if keep_missing:
        for k in sorted(all_keys - inter_keys):
            # choose the first model that has it
            for s in ckpts:
                if k in s:
                    v = s[k]
                    out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
                    break

    # Ensure everything is CPU tensors (common expectation for a raw state_dict)
    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor) and v.device.type != "cpu":
            out[k] = v.cpu()

    return out


def average_state_dicts(
    model_dirs,
    skip_key_pred=None,
    dtype_str: Literal["float32", "bfloat16"] = "bfloat16",
    mode="arithmetic",
    force_model_type: Optional[str] = None,
):
    assert len(model_dirs) > 1, "Need at least two models to average"
    assert dtype_str in ["float32", "bfloat16"], "dtype must be 'float32' or 'bfloat16'"
    """
    Average state_dicts across models.
    - mode='arithmetic' (standard). 'harmonic' supported but rarely sensible for signed weights.
    - skip_key_pred: function(key) -> bool to skip some keys (e.g. heads).
    """
    sums = defaultdict(torch.Tensor)
    counts = defaultdict(int)
    template_sd = None

    dtype = torch.float32 if dtype_str == "float32" else torch.bfloat16

    for ix, d in enumerate(model_dirs):
        sd = load_state_dict(d, force_model_type=force_model_type)
        if template_sd is None:
            template_sd = sd  # keep keys/layout from the first model
        for k, v in sd.items():
            if skip_key_pred and skip_key_pred(k):
                continue
            if torch.is_floating_point(v):
                t = v.to(dtype)
                if k not in sums:
                    sums[k] = t.clone()
                    counts[k] = 1
                else:
                    if mode == "arithmetic":
                        sums[k].add_(t)
                        counts[k] += 1
                    elif mode == "harmonic":
                        # Harmonic mean: n / sum(1/x); protect against zeros
                        eps = 1e-12
                        inv = 1.0 / (t.abs() + eps)
                        # Store accumulator as sum of inverses; reconstruct later
                        if counts[k] == 0:
                            sums[k] = inv
                        else:
                            sums[k].add_(inv)
                        counts[k] += 1
                    else:
                        raise ValueError(f"Unknown mode: {mode}")
            # non-floating keys are handled later (kept from template)
        # free sd asap
        del sd

    # Build averaged sd using template keys/layout
    averaged = {}
    for k, v in template_sd.items():
        if skip_key_pred and skip_key_pred(k):
            # Keep from first model (or skip entirely; here we keep to remain loadable)
            averaged[k] = v
            continue
        if torch.is_floating_point(v) and k in sums and counts[k] > 0:
            if mode == "arithmetic":
                averaged_val = (sums[k] / counts[k]).to(v.dtype)
            else:  # harmonic
                n = counts[k]
                eps = 1e-12
                averaged_val = (n / (sums[k] + eps)).to(v.dtype)
            averaged[k] = averaged_val
        else:
            # ints/bools/buffers: keep from first model
            averaged[k] = v
    return averaged


def model_averager(model_locations, mode="arithmetic", force_model_type: Optional[str] = None):
    # Load the first model to get initial weights
    model = load_token_cls_model(
        model_locations[0], force_model_type=force_model_type
    )
    averaged_weights = {
        name: torch.zeros_like(param) for name, param in model.named_parameters()
    }
    num_models = len(model_locations)

    # Iterate through each model, accumulate weights
    for location in model_locations:
        print(f"Processing model at {location}")
        model = load_token_cls_model(
            location, force_model_type=force_model_type
        )
        for name, param in model.named_parameters():
            if mode == "arithmetic":
                averaged_weights[name] += param.data
            elif mode == "harmonic":
                # For harmonic mean: accumulate reciprocals (1/x), protect against zeros
                eps = 1e-12
                averaged_weights[name] += 1.0 / (param.data.abs() + eps)

    # Perform averaging based on the number of models and mode
    for name, param in averaged_weights.items():
        if mode == "arithmetic":
            averaged_weights[name] /= num_models
        elif mode == "harmonic":
            # Harmonic mean: n / sum(1/x)
            averaged_weights[name] = num_models / averaged_weights[name]

    # Load a new model (or reuse the first model) and update weights
    model_averaged = load_token_cls_model(
        model_locations[0], force_model_type=force_model_type
    )
    model_averaged.load_state_dict(averaged_weights, strict=False)

    # Handle the heads separately if needed
    head_averaged_weights = {}
    for location in model_locations:
        model = load_token_cls_model(location, force_model_type=force_model_type)
        for name, param in model.classifier.named_parameters():
            print(f"Appending layer: {name}, of shape {param.shape}", flush=True)
            if name not in head_averaged_weights:
                print(f"Init {name}")
                head_averaged_weights[name] = torch.zeros_like(param)

            if mode == "arithmetic":
                head_averaged_weights[name] += param.data
            elif mode == "harmonic":
                # For harmonic mean: accumulate reciprocals (1/x), protect against zeros
                eps = 1e-12
                head_averaged_weights[name] += 1.0 / (param.data.abs() + eps)

    for name, param in head_averaged_weights.items():
        print(f"Averaging layer: {name}", flush=True)
        if mode == "arithmetic":
            head_averaged_weights[name] /= num_models
        elif mode == "harmonic":
            # Harmonic mean: n / sum(1/x)
            head_averaged_weights[name] = num_models / head_averaged_weights[name]

    # Update the head weights
    model_averaged.classifier.load_state_dict(head_averaged_weights, strict=False)

    return model_averaged.state_dict()


def copy_model_metadata_and_code(source_model_dir: str, output_dir: str) -> None:
    """
    Preserve source model metadata/code files that `save_pretrained` may rewrite or omit.

    - Restores the original `config.json` from the source model directory.
    - Copies custom python code files referenced by `auto_map` in config.
    - Also copies a conventional `modeling.py` if present.
    """
    os.makedirs(output_dir, exist_ok=True)

    source_config_path = os.path.join(source_model_dir, "config.json")
    output_config_path = os.path.join(output_dir, "config.json")

    source_cfg = {}
    if os.path.isfile(source_config_path):
        shutil.copy2(source_config_path, output_config_path)
        try:
            with open(source_config_path, "r", encoding="utf-8") as f:
                source_cfg = json.load(f)
        except Exception as e:
            print(
                f"[metadata-copy] Could not parse source config.json from {source_model_dir}: {e}",
                flush=True,
            )

    files_to_copy: Set[str] = set()

    # If auto_map exists, copy referenced module files such as "modeling_xxx.py".
    auto_map = source_cfg.get("auto_map") if isinstance(source_cfg, dict) else None
    if isinstance(auto_map, dict):
        for v in auto_map.values():
            values = v if isinstance(v, list) else [v]
            for item in values:
                if not isinstance(item, str):
                    continue
                # Typical shapes:
                # - "module.ClassName"
                # - "repo-id--module.ClassName"
                module_spec = item.split("--")[-1]
                module_name = module_spec.split(".")[0].strip()
                if module_name:
                    files_to_copy.add(f"{module_name}.py")

    # Conventional fallback used by many custom HF models.
    files_to_copy.add("modeling.py")

    for filename in sorted(files_to_copy):
        src = os.path.join(source_model_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def path_parser(models_dir):
    """
    Parses a given directory to find all model paths.

    Args:
      models_dir: The directory containing the model files or folders.

    Returns:
      A list of paths to all the models found within the directory.
    """
    model_paths = []
    for root, dirs, files in os.walk(models_dir):
        # if last folder name contains 'average' skip
        if "average" in os.path.basename(root):
            continue
        for file in files:
            if (
                ("model" in file)
                and (file.endswith(".bin") or file.endswith(".safetensors"))
            ) or os.path.isdir(os.path.join(root, file)):
                model_paths.append(os.path.join(root))
                continue
    return model_paths


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(exit_on_error=True)
    argparser.add_argument("--models_dir", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument(
        "--averaging",
        type=str,
        choices=["arithmetic", "harmonic", "chordal", "karcher", "chain"],
        default="arithmetic",
    )
    argparser.add_argument("--version", type=int, default=1)
    argparser.add_argument(
        "--dtype", type=str, choices=["float32", "bfloat16"], default="bfloat16"
    )
    argparser.add_argument(
        "--force_model_type",
        type=str,
        default=None,
        help=(
            "Force model family when checkpoint config is wrong/mislabeled. "
            "Supported: bert, roberta (alias: robert)."
        ),
    )

    args = argparser.parse_args()
    args.force_model_type = normalize_model_type(args.force_model_type)

    list_of_model_locations = path_parser(args.models_dir)
    print("Models to merge..")
    print(list_of_model_locations)

    model_config = load_model_config(
        list_of_model_locations[0], force_model_type=args.force_model_type
    )

    if args.averaging in ["chordal", "karcher", "chain"]:
        args.version = 3
        print(f"Continuing with SLERP averaging: {args.averaging}", flush=True)

    if args.version == 1:
        print(
            f"Performing {args.averaging} averaging using method {args.version}...",
            flush=True,
        )
        new_state_dict = model_averager(
            list_of_model_locations,
            mode=args.averaging,
            force_model_type=args.force_model_type,
        )
    elif args.version == 2:
        print(
            f"Performing {args.averaging} averaging using method {args.version}...",
            flush=True,
        )
        new_state_dict = average_state_dicts(
            list_of_model_locations,
            dtype_str=args.dtype,
            mode=args.averaging,
            force_model_type=args.force_model_type,
        )
    elif args.version == 3:
        print(
            f"Performing {args.averaging} averaging using method {args.version}...",
            flush=True,
        )
        new_state_dict = average_state_dict_advanced(
            list_of_model_locations,
            method=args.averaging,
            force_model_type=args.force_model_type,
        )
    else:
        raise ValueError("Invalid version: has to be one of [1,2]")

    if args.force_model_type is None:
        model_averaged = transformers.AutoModelForTokenClassification.from_config(
            config=model_config, trust_remote_code=True
        )
    else:
        _, forced_model_cls, _ = SUPPORTED_FORCED_MODEL_TYPES[args.force_model_type]
        model_averaged = forced_model_cls(model_config)
    missing, unexpected = model_averaged.load_state_dict(new_state_dict, strict=False)

    if missing:
        print("[load_state_dict] Missing keys:", missing)
    if unexpected:
        print("[load_state_dict] Unexpected keys:", unexpected)

    if args.dtype == "float32":
        model_averaged = model_averaged.to(torch.float32)
    elif args.dtype == "bfloat16":
        model_averaged = model_averaged.to(torch.bfloat16)
    else:
        raise ValueError("dtype must be 'float32' or 'bfloat16'")

    os.makedirs(args.output_dir, exist_ok=True)
    model_averaged.save_pretrained(args.output_dir)

    # Preserve source config/custom code files that can be changed/omitted by save_pretrained.
    copy_model_metadata_and_code(
        source_model_dir=list_of_model_locations[0], output_dir=args.output_dir
    )
