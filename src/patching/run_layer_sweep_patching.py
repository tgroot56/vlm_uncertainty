"""Layer-sweep activation patching experiment for final prompt token positions.

For each sample, extracts clean activations at ALL LM layers for the final
``n_prompt_tokens`` non-visual prompt positions (default 3), then for every
(layer, position) pair runs the degraded forward pass with that single
activation patched in.

This answers: at which LM layer does the representation of the final prompt
token carry enough clean context to restore model confidence / accuracy?

Outputs
-------
- ``layer_sweep_results.parquet``  — per-sample raw rows
- ``layer_sweep_summary.csv``      — mean recovery per (severity, layer, position_offset)
- ``layer_sweep_recovery_*.pdf/png`` — recovery curves, x=layer, y=recovery
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from src.patching.collect_activations import (
    VERSION_CLEAN,
    VISUAL_SEVERITY_VERSIONS,
    _format_prompt,
    _MAX_NEW_TOKENS,
    _score_prediction,
)
from src.patching.run_patching import (
    _compute_output_stats,
    _get_decoder_layers,
    _patched_generate,
    _run_baseline_forward,
)
from src.uq_dataset_generation.token_spans import build_token_spans
from src.vlm import VLMAdapter, prepare_inputs, prepare_qwen_fused_inputs_embeds

RECOVERY_EPSILON = 1e-8
RECOVERY_CLIP = 5.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LayerSweepConfig:
    """Configuration for a layer-sweep patching run."""
    dataset_id: str = "vqa-v2"
    n_prompt_tokens: int = 3       # how many final prompt tokens to patch
    severity: str = "moderate"     # "mild", "moderate", "severe", or "all"
    max_new_tokens: int = 4
    checkpoint_every: int = 10
    verbose: bool = False
    # "final" (default): last ``n_prompt_tokens`` non-visual prompt tokens.
    # "question_visual": exactly two positions — last question token
    # (``question_end - 1``) and last LM visual token (``visual_end - 1``).
    position_set: str = "final"
    # When set, write a snapshot parquet + plots into
    # ``<output_dir>/intermediate_at_<N>/`` once ``N`` samples have completed.
    # The main run continues unaffected.
    plot_at_n_samples: Optional[int] = None
    # Qwen-only: append an empty ``<think></think>`` block to the prompt so
    # generated step 0 is the first answer token.
    qwen_prefill_empty_thinking: bool = False


# ---------------------------------------------------------------------------
# Clean activation extraction — all layers at once
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_all_layer_activations(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    image: Image.Image,
    prompt: str,
    qwen_prefill_empty_thinking: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Single clean forward pass to capture decoder-block outputs.

    Returns
    -------
    all_layer_h : tensor of shape ``(num_decoder_layers, prompt_len, H)`` on CPU.
        Index ``k`` is the actual output of decoder block ``k``. This matches
        the hook site used by ``_patched_generate`` exactly. Do not use
        ``outputs.hidden_states`` here: HF-style models often include a
        post-final-norm readout state as the last hidden-state entry, which is
        not the same activation space as the final decoder-block output.
    token_spans : dict from ``build_token_spans()``.
    """
    inputs = prepare_inputs(
        vlm_adapter,
        processor,
        image,
        prompt,
        device,
        qwen_prefill_empty_thinking=qwen_prefill_empty_thinking,
    )
    input_ids = inputs["input_ids"][0]
    prompt_len = int(inputs["input_ids"].shape[1])

    decoder_layers = _get_decoder_layers(model, vlm_adapter)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _args, output) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states.ndim != 3:
                raise RuntimeError(
                    f"Decoder layer {layer_idx} returned hidden states with "
                    f"shape {tuple(hidden_states.shape)}, expected rank 3."
                )
            captured[layer_idx] = hidden_states[0].detach().cpu()

        return _hook

    for layer_idx, layer in enumerate(decoder_layers):
        handles.append(layer.register_forward_hook(_make_hook(layer_idx)))

    try:
        model(**inputs, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    missing = [idx for idx in range(len(decoder_layers)) if idx not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture decoder-block outputs for layers: {missing}")

    all_layer_h = torch.stack(
        [captured[idx] for idx in range(len(decoder_layers))], dim=0
    )
    if int(all_layer_h.shape[1]) != prompt_len:
        raise RuntimeError(
            f"Captured decoder activations have seq_len={all_layer_h.shape[1]}, "
            f"but prompt_len={prompt_len}."
        )

    lm_hidden_len = int(all_layer_h.shape[1])
    token_spans = build_token_spans(
        model=model,
        processor=processor,
        vlm_adapter=vlm_adapter,
        input_ids=input_ids,
        prompt_len=prompt_len,
        lm_hidden_len=lm_hidden_len,
    )

    return all_layer_h, token_spans


def _get_layer_index_pairs(
    num_hidden_state_layers: int,
    num_decoder_layers: int,
) -> List[Tuple[int, int]]:
    """Map captured activation indices to patchable decoder-layer indices.

    The corrected layer sweep captures decoder-block outputs directly, so the
    common case is identity mapping: captured index ``k`` patches decoder block
    ``k``. The ``+1`` case is retained for older callers that pass HF
    ``hidden_states`` tuples with an embedding slot at index 0.
    """
    if num_hidden_state_layers == num_decoder_layers + 1:
        return [
            (hidden_state_idx, hidden_state_idx - 1)
            for hidden_state_idx in range(1, num_hidden_state_layers)
        ]
    if num_hidden_state_layers == num_decoder_layers:
        return [
            (hidden_state_idx, hidden_state_idx)
            for hidden_state_idx in range(num_hidden_state_layers)
        ]
    raise RuntimeError(
        "Could not align LM hidden states with decoder layers: "
        f"hidden_state_layers={num_hidden_state_layers}, "
        f"decoder_layers={num_decoder_layers}."
    )


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def _get_final_prompt_positions(
    token_spans: Dict[str, Any],
    n: int,
) -> List[int]:
    """Return the last ``n`` non-visual token positions before generation.

    The full prompt span ends at ``fullspan_end`` (= ``prompt_len``).  We
    skip any visual tokens by walking backwards from the end of the prompt,
    returning up to ``n`` positions from the non-visual region.
    """
    v_start = token_spans["visual_start"]
    v_end = token_spans["visual_end"]
    prompt_len = token_spans["prompt_len"]

    non_visual = list(range(0, v_start)) + list(range(v_end, prompt_len))
    return non_visual[-n:]  # last n, in ascending order


def _position_offset_label(pos: int, token_spans: Dict[str, Any]) -> int:
    """Return the offset from the end of the prompt (−1 = last token, etc.)."""
    return pos - token_spans["prompt_len"]


def _get_question_visual_positions(
    token_spans: Dict[str, Any],
) -> Tuple[List[int], List[str]]:
    """Return two positions: last question token and last LM visual token.

    Both ends are exclusive in ``token_spans`` (…``_end``), so the last
    token of a span is ``*_end - 1``. Returns ``(positions, labels)`` with
    labels aligned (``"last_question"``, ``"last_visual"``).
    """
    positions: List[int] = []
    labels: List[str] = []

    q_end = int(token_spans["question_end"])
    q_start = int(token_spans["question_start"])
    if q_end > q_start:
        positions.append(q_end - 1)
        labels.append("last_question")

    v_end = int(token_spans["visual_end"])
    v_start = int(token_spans["visual_start"])
    if v_end > v_start:
        positions.append(v_end - 1)
        labels.append("last_visual")

    return positions, labels


# ---------------------------------------------------------------------------
# Checkpoint / save helpers
# ---------------------------------------------------------------------------

def _save_results(
    rows: List[Dict[str, Any]],
    output_dir: str,
    checkpoint_path: Optional[str],
    completed_ids: set,
) -> None:
    import pandas as pd

    parquet_path = os.path.join(output_dir, "layer_sweep_results.parquet")
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    if checkpoint_path is not None:
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"completed_ids": sorted(completed_ids)}, f)
        os.replace(tmp, checkpoint_path)


def _build_layer_sweep_recovery_table(df):
    """Attach per-sample recovery columns to layer-sweep patched rows."""
    import numpy as np

    clean = (
        df[df["condition"] == "clean"]
        .drop_duplicates(["sample_id", "severity"])
        [[
            "sample_id",
            "severity",
            "score",
            "top1_probability",
            "output_entropy",
        ]]
        .rename(
            columns={
                "score": "score_clean",
                "top1_probability": "top1_probability_clean",
                "output_entropy": "output_entropy_clean",
            }
        )
    )
    degraded = (
        df[df["condition"] == "degraded"]
        .drop_duplicates(["sample_id", "severity"])
        [[
            "sample_id",
            "severity",
            "score",
            "top1_probability",
            "output_entropy",
        ]]
        .rename(
            columns={
                "score": "score_degraded",
                "top1_probability": "top1_probability_degraded",
                "output_entropy": "output_entropy_degraded",
            }
        )
    )

    patched = df[df["condition"] == "patched"].copy()
    patched = patched.merge(clean, on=["sample_id", "severity"], how="left")
    patched = patched.merge(degraded, on=["sample_id", "severity"], how="left")

    metric_specs = [
        ("score", "recovery_score"),
        ("top1_probability", "recovery_top1"),
        ("output_entropy", "recovery_entropy"),
    ]
    for metric, recovery_col in metric_specs:
        num = patched[metric] - patched[f"{metric}_degraded"]
        den = patched[f"{metric}_clean"] - patched[f"{metric}_degraded"] + RECOVERY_EPSILON
        patched[recovery_col] = np.clip(num / den, -RECOVERY_CLIP, RECOVERY_CLIP)

    return patched


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_layer_sweep_patching(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    vlm_adapter: VLMAdapter,
    patching_records: List[Dict[str, Any]],
    output_dir: str,
    cfg: LayerSweepConfig,
) -> None:
    """Execute layer-sweep patching and save per-sample results.

    For every sample:
      1. Forward pass on the clean image → collect hidden states at ALL layers
         for the final ``cfg.n_prompt_tokens`` non-visual token positions.
      2. Forward pass on each degraded image → degraded baseline.
      3. For each (layer, position) pair: patched forward pass on the degraded
         image with one token's activation replaced at one layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    max_new_tokens = _MAX_NEW_TOKENS.get(cfg.dataset_id, cfg.max_new_tokens)

    severities = (
        ["mild", "moderate", "severe", "extreme"] if cfg.severity == "all"
        else [cfg.severity]
    )

    checkpoint_path = os.path.join(output_dir, "layer_sweep_checkpoint.json")
    completed_ids: set = set()
    rows: List[Dict[str, Any]] = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_ids = {str(sample_id) for sample_id in ckpt.get("completed_ids", [])}
        parquet_path = os.path.join(output_dir, "layer_sweep_results.parquet")
        if completed_ids and os.path.exists(parquet_path):
            import pandas as pd
            rows = pd.read_parquet(parquet_path).to_dict("records")
        print(f"[layer-sweep] Resuming: {len(completed_ids)} samples done.")
    else:
        parquet_path = os.path.join(output_dir, "layer_sweep_results.parquet")
        if os.path.exists(parquet_path):
            import pandas as pd
            existing = pd.read_parquet(parquet_path)
            if not existing.empty and "sample_id" in existing.columns:
                rows = existing.to_dict("records")
                completed_ids = {
                    str(sample_id)
                    for sample_id in existing["sample_id"].dropna().unique().tolist()
                }
                print(
                    f"[layer-sweep] Resuming from existing parquet: "
                    f"{len(completed_ids)} samples done."
                )

    total = len(patching_records)
    print(f"[layer-sweep] Starting layer-sweep patching")
    print(f"  Samples:          {total}")
    print(f"  Final positions:  {cfg.n_prompt_tokens}")
    print(f"  Severities:       {severities}")
    print(f"  Qwen prefill:     {cfg.qwen_prefill_empty_thinking}")
    print(f"  Output:           {output_dir}/")

    t0 = time.time()

    for record in tqdm(patching_records, desc="layer-sweep patching"):
        sample_id = str(record["sample_id"])
        if sample_id in completed_ids:
            continue

        prompt = _format_prompt(cfg.dataset_id, record["clean_question"])
        clean_image = Image.open(record["clean_image_path"]).convert("RGB")

        # ---- clean forward pass: all layers --------------------------------
        all_layer_h, token_spans = _extract_all_layer_activations(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
        # all_layer_h: (num_decoder_layers, prompt_len, H)
        decoder_layers = _get_decoder_layers(model, vlm_adapter)
        layer_index_pairs = _get_layer_index_pairs(
            num_hidden_state_layers=all_layer_h.shape[0],
            num_decoder_layers=len(decoder_layers),
        )

        if cfg.position_set == "question_visual":
            positions, span_labels = _get_question_visual_positions(token_spans)
        elif cfg.position_set == "final":
            positions = _get_final_prompt_positions(token_spans, cfg.n_prompt_tokens)
            span_labels = [None] * len(positions)
        else:
            raise ValueError(
                f"Unknown position_set '{cfg.position_set}'. "
                "Use 'final' or 'question_visual'."
            )
        offsets = [_position_offset_label(p, token_spans) for p in positions]

        if not positions:
            if cfg.verbose:
                print(f"  [skip] {sample_id}: no patchable positions for position_set={cfg.position_set}")
            continue

        if cfg.verbose:
            print(
                f"\n  [sample {sample_id}] prompt_len={token_spans['prompt_len']}, "
                f"decoder_layers={len(layer_index_pairs)}, "
                f"patching positions={positions} (offsets={offsets})"
            )

        # ---- Qwen fast path: pre-compute fused embeds once per image.
        # Vision tower is ~30-40x slower than the rest of one patching call
        # on Qwen, and it doesn't depend on the patched hidden state, so we
        # run it once per (image, prompt) and reuse across all layer×position
        # conditions. Non-Qwen VLMs keep the original path.
        use_fused = vlm_adapter.family == "qwen"
        clean_fused = (
            prepare_qwen_fused_inputs_embeds(
                model,
                vlm_adapter,
                processor,
                clean_image,
                prompt,
                device,
                qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
            )
            if use_fused else None
        )

        # ---- clean baseline (unpatched) ------------------------------------
        clean_fwd = _run_baseline_forward(
            model=model, processor=processor, device=device,
            vlm_adapter=vlm_adapter, image=clean_image, prompt=prompt,
            max_new_tokens=max_new_tokens, fused_bundle=clean_fused,
            qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
        )
        clean_image.close()
        clean_stats = _compute_output_stats(
            clean_fwd["gen_step_logits"], clean_fwd.get("gen_ids"), processor
        )
        clean_score = _score_prediction(cfg.dataset_id, clean_fwd["raw_text"], record)

        # ---- per-severity loop --------------------------------------------
        for pert in record.get("visual_perturbations", []):
            severity = pert["severity"]
            if severity not in severities:
                continue

            img_path = pert.get("blurred_image_path", "")
            if not img_path or not os.path.exists(img_path):
                continue

            blurred_image = Image.open(img_path).convert("RGB")
            blurred_fused = (
                prepare_qwen_fused_inputs_embeds(
                    model,
                    vlm_adapter,
                    processor,
                    blurred_image,
                    prompt,
                    device,
                    qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
                )
                if use_fused else None
            )

            # Degraded baseline
            deg_fwd = _run_baseline_forward(
                model=model, processor=processor, device=device,
                vlm_adapter=vlm_adapter, image=blurred_image, prompt=prompt,
                max_new_tokens=max_new_tokens, fused_bundle=blurred_fused,
                qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
            )
            deg_stats = _compute_output_stats(
                deg_fwd["gen_step_logits"], deg_fwd.get("gen_ids"), processor
            )
            deg_score = _score_prediction(cfg.dataset_id, deg_fwd["raw_text"], record)

            # Record baselines
            for baseline_cond, fwd, stats, score in [
                ("clean",   clean_fwd,  clean_stats, clean_score),
                ("degraded", deg_fwd,  deg_stats,   deg_score),
            ]:
                rows.append({
                    "sample_id": sample_id,
                    "severity": severity,
                    "condition": baseline_cond,
                    "layer": None,
                    "token_position": None,
                    "position_offset": None,
                    "span_label": None,
                    "predicted_answer": fwd["raw_text"],
                    "score": score,
                    **stats,
                })

            # ---- layer × position sweep -----------------------------------
            for hidden_state_idx, patch_layer_idx in layer_index_pairs:
                for pos, offset, span_label in zip(positions, offsets, span_labels):
                    clean_act = all_layer_h[hidden_state_idx, pos, :].unsqueeze(0)  # (1, H)

                    patched_fwd = _patched_generate(
                        model=model, processor=processor, device=device,
                        vlm_adapter=vlm_adapter, image=blurred_image,
                        prompt=prompt, max_new_tokens=max_new_tokens,
                        layer_idx=patch_layer_idx,
                        positions=[pos],
                        clean_activations=clean_act,
                        fused_bundle=blurred_fused,
                        qwen_prefill_empty_thinking=cfg.qwen_prefill_empty_thinking,
                    )
                    patched_stats = _compute_output_stats(
                        patched_fwd["gen_step_logits"], patched_fwd.get("gen_ids"), processor
                    )
                    patched_score = _score_prediction(
                        cfg.dataset_id, patched_fwd["raw_text"], record,
                    )

                    rows.append({
                        "sample_id": sample_id,
                        "severity": severity,
                        "condition": "patched",
                        "layer": patch_layer_idx,
                        "token_position": pos,
                        "position_offset": offset,
                        "span_label": span_label,
                        "predicted_answer": patched_fwd["raw_text"],
                        "score": patched_score,
                        **patched_stats,
                    })

                    if cfg.verbose:
                        print(
                            f"    layer={patch_layer_idx} offset={offset} pos={pos} "
                            f"deg_top1={deg_stats['top1_probability']:.3f} "
                            f"→ patched_top1={patched_stats['top1_probability']:.3f} "
                            f"(clean={clean_stats['top1_probability']:.3f})"
                        )

            blurred_image.close()

        completed_ids.add(sample_id)

        if len(completed_ids) % cfg.checkpoint_every == 0:
            _save_results(rows, output_dir, checkpoint_path, completed_ids)
            elapsed = time.time() - t0
            rate = len(completed_ids) / elapsed if elapsed > 0 else 0
            print(
                f"[layer-sweep] {len(completed_ids)}/{total} samples "
                f"({elapsed:.0f}s, {rate:.2f} samples/s)"
            )

        if (
            cfg.plot_at_n_samples is not None
            and len(completed_ids) == cfg.plot_at_n_samples
        ):
            _write_intermediate_snapshot(
                rows=rows,
                output_dir=output_dir,
                n_samples=len(completed_ids),
            )

    # Final save
    _save_results(rows, output_dir, checkpoint_path=None, completed_ids=completed_ids)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - t0
    print(f"\n[layer-sweep] Finished {len(completed_ids)}/{total} samples in {elapsed:.1f}s")

    _print_layer_sweep_summary(rows, output_dir)

    from pathlib import Path as _Path
    from src.cli.plot_sweep_filtered import plot_layer_sweep_combined

    parquet_path = os.path.join(output_dir, "layer_sweep_results.parquet")
    try:
        plot_layer_sweep_combined(parquet_path, _Path(output_dir), raw=False)
        plot_layer_sweep_combined(parquet_path, _Path(output_dir), raw=True)
        plot_layer_sweep_combined(parquet_path, _Path(output_dir), raw=True, min_delta=0.0)
    except Exception as e:
        print(f"[layer-sweep] Plot step failed ({type(e).__name__}: {e}); parquet written OK.")


# ---------------------------------------------------------------------------
# Intermediate snapshot
# ---------------------------------------------------------------------------

def _write_intermediate_snapshot(
    *,
    rows: List[Dict[str, Any]],
    output_dir: str,
    n_samples: int,
) -> None:
    """Write a parquet + plots snapshot to ``<output_dir>/intermediate_at_<N>/``.

    Used to give a quick read on the in-progress run without disturbing the
    final outputs. The main loop continues unaffected.
    """
    import pandas as pd
    from pathlib import Path as _Path
    from src.cli.plot_sweep_filtered import plot_layer_sweep_combined

    snap_dir = os.path.join(output_dir, f"intermediate_at_{n_samples}")
    os.makedirs(snap_dir, exist_ok=True)
    snap_parquet = os.path.join(snap_dir, "layer_sweep_results.parquet")
    pd.DataFrame(rows).to_parquet(snap_parquet, index=False)
    print(f"[layer-sweep] Wrote intermediate snapshot to {snap_dir}/")

    try:
        plot_layer_sweep_combined(snap_parquet, _Path(snap_dir), raw=False)
        plot_layer_sweep_combined(snap_parquet, _Path(snap_dir), raw=True)
        plot_layer_sweep_combined(snap_parquet, _Path(snap_dir), raw=True, min_delta=0.0)
    except Exception as e:
        print(
            f"[layer-sweep] Intermediate plot step failed "
            f"({type(e).__name__}: {e}); snapshot parquet written OK."
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_layer_sweep_summary(
    rows: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Compute per-(layer, position_offset) recovery and save summary CSV."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        print("[layer-sweep] No results to summarize.")
        return

    patched_with_recovery = _build_layer_sweep_recovery_table(df)
    severities = df[df["condition"] == "degraded"]["severity"].dropna().unique()
    summary_rows: List[Dict[str, Any]] = []

    for severity in severities:
        sev_df = df[df["severity"] == severity]
        sev_patched = patched_with_recovery[patched_with_recovery["severity"] == severity]

        agg = sev_patched.groupby(["layer", "position_offset"]).agg(
            mean_score        =("score",           "mean"),
            mean_top1_prob    =("top1_probability", "mean"),
            mean_entropy      =("output_entropy",   "mean"),
            recovery_score    =("recovery_score",   "mean"),
            recovery_top1     =("recovery_top1",    "mean"),
            recovery_entropy  =("recovery_entropy", "mean"),
            N                 =("sample_id",        "count"),
        ).reset_index()

        for _, row in agg.iterrows():
            summary_rows.append({
                "severity":         severity,
                "layer":            int(row["layer"]),
                "position_offset":  int(row["position_offset"]),
                "mean_score":       row["mean_score"],
                "mean_top1_prob":   row["mean_top1_prob"],
                "mean_entropy":     row["mean_entropy"],
                "recovery_score":   row["recovery_score"],
                "recovery_top1":    row["recovery_top1"],
                "recovery_entropy": row["recovery_entropy"],
                "N":                int(row["N"]),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "layer_sweep_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[layer-sweep] Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_layer_sweep_recovery(
    results_parquet: str,
    output_dir: str,
    *,
    title_prefix: str = "",
) -> None:
    """Plot layer-sweep recovery curves.

    Three vertically-stacked subplots:
      - Score recovery
      - Top-1 probability recovery
      - Entropy recovery

    x-axis = decoder block index (0 = output of transformer block 0)
    y-axis = recovery metric

    One curve per (position_offset, severity) combination.
    position_offset −1 = last prompt token, −2 = second-to-last, etc.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if not os.path.exists(results_parquet):
        print(f"[layer-sweep] No results file at {results_parquet}, skipping plot.")
        return

    df = pd.read_parquet(results_parquet)
    if df.empty:
        print("[layer-sweep] Empty results, skipping plot.")
        return

    severities = sorted(df[df["condition"] == "degraded"]["severity"].dropna().unique())
    patched_with_recovery = _build_layer_sweep_recovery_table(df)

    # Position offsets present in the data, sorted (most negative = earliest token)
    offsets = sorted(
        df[df["condition"] == "patched"]["position_offset"].dropna().unique().astype(int)
    )

    # Colour by position offset; line style by severity
    OFFSET_COLORS = {
        -3: "#7c3aed",  # purple   – third-to-last
        -2: "#0891b2",  # cyan     – second-to-last
        -1: "#16a34a",  # green    – last
    }
    # Fall back to a generic palette for other offsets
    _FALLBACK = ["#f59e0b", "#ef4444", "#0ea5e9", "#9ca3af"]

    def _offset_color(offset: int) -> str:
        return OFFSET_COLORS.get(offset, _FALLBACK[abs(offset) % len(_FALLBACK)])

    SEVERITY_STYLES = {
        "mild":     ("--", 0.55),
        "moderate": ("-",  0.85),
        "severe":   (":",  0.70),
        "extreme":  ((0, (3, 1, 1, 1)), 0.95),
    }

    metrics = [
        ("score",           "top1_probability", "output_entropy"),
        ("Score Recovery",  "Top-1 Probability Recovery", "Entropy Recovery"),
    ]
    raw_cols    = ["score",           "top1_probability", "output_entropy"]
    metric_keys = ["recovery_score",  "recovery_top1",    "recovery_entropy"]
    metric_labels = ["Score Recovery", "Top-1 Prob Recovery", "Entropy Recovery"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    for severity in severities:
        patch_df = patched_with_recovery[patched_with_recovery["severity"] == severity]

        linestyle, alpha = SEVERITY_STYLES.get(severity, ("-", 0.8))

        for offset in offsets:
            off_df = patch_df[patch_df["position_offset"] == offset]
            layer_agg = off_df.groupby("layer").agg(
                recovery_score    =("recovery_score",   "mean"),
                recovery_top1     =("recovery_top1",    "mean"),
                recovery_entropy  =("recovery_entropy", "mean"),
            ).sort_index()

            layers = layer_agg.index.values
            color = _offset_color(offset)
            label = f"offset {offset:+d}, {severity}"

            recovery_cols = ["recovery_score", "recovery_top1", "recovery_entropy"]

            for ax, recovery_col in zip(axes, recovery_cols):
                recovery = layer_agg[recovery_col].values
                ax.plot(
                    layers, recovery,
                    color=color, linestyle=linestyle, linewidth=1.6,
                    alpha=alpha, label=label,
                    marker="", zorder=3,
                )

    for ax, title in zip(axes, metric_labels):
        ax.axhline(y=0.0, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(y=1.0, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.15)

    axes[-1].set_xlabel("Layer index", fontsize=11)

    # ---- legend ----
    # Build a compact legend: one entry per (offset, severity)
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate while preserving order
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    axes[0].legend(
        seen.values(), seen.keys(),
        loc="upper left", fontsize=8, ncol=2,
        framealpha=0.85,
    )

    title_str = f"{title_prefix}Layer-Sweep Recovery — Final {len(offsets)} Prompt Tokens"
    fig.suptitle(title_str, fontsize=13, y=1.005)
    fig.tight_layout()

    sev_tag = "all" if len(severities) > 1 else severities[0]
    for ext in ("pdf", "png"):
        fname = f"layer_sweep_recovery_{sev_tag}.{ext}"
        save_kwargs = {"dpi": 150} if ext == "png" else {}
        fig.savefig(
            os.path.join(output_dir, fname),
            bbox_inches="tight", **save_kwargs,
        )

    plt.close(fig)
    print(
        f"[layer-sweep] Plot saved to "
        f"{output_dir}/layer_sweep_recovery_{sev_tag}.pdf"
    )
