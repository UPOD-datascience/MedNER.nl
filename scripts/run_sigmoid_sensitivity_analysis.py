#!/usr/bin/env python
"""
Span-level decoding ablation: softmax vs. sigmoid confidence scores.

Motivation
----------
`cardioner.predictor.PredictionNER` converts token logits to confidences with
`F.softmax` before applying the decoding thresholds (the 0.70 O-class fallback
and the 0.65 / 0.10 entity-acceptance thresholds). That is the correct scale for
multiclass models, but multilabel models are trained with `BCEWithLogitsLoss`,
i.e. each label is an independent sigmoid decision. Softmax redistributes
probability mass across co-activated labels, so multilabel predictions are
scored on a scale they were never calibrated for and are filtered more
aggressively than multiclass ones.

This script re-runs the *existing* fold inference pipeline with the confidence
scale swapped to sigmoid, changing nothing else. Label selection is provably
unaffected: `predict_text_batch` takes `torch.argmax` over the raw logits, not
over `probs`, and argmax is invariant under both transforms. Only the confidence
values that feed the thresholds change -- so any delta is attributable to
threshold calibration alone.

Usage
-----
    python run_sigmoid_ablation.py \
        --activation sigmoid \
        --bulk_file   .../annotations_all_entities.jsonl \
        --split_file  .../dataset_splits/splits_cv10_holdout_test.json \
        --model_folder .../multilabel_cardioberta/cz_all_entities \
        --output_dir  .../cz_all_entities/inference_eval_sigmoid \
        --lang cz \
        --pipe dt4h

Run with `--activation softmax` to reproduce the original numbers through this
same wrapper (useful as a control: it should match the reported results exactly).

Point `--output_dir` at a NEW directory so existing results are not overwritten.
"""

import argparse
import sys

import torch


def patch_predictor_to_sigmoid() -> None:
    """
    Rebind `cardioner.predictor.F` so that `F.softmax(...)` yields sigmoid
    probabilities. Every other attribute is delegated to torch.nn.functional.

    Scope: this touches only the `F` name inside the `cardioner.predictor`
    module namespace. `torch.nn.functional` itself is untouched, so no other
    module's behaviour changes.
    """
    import torch.nn.functional as _F

    import cardioner.predictor as predictor

    class _SigmoidFunctional:
        """Drop-in stand-in for torch.nn.functional with softmax -> sigmoid."""

        @staticmethod
        def softmax(input, dim=-1, *args, **kwargs):  # noqa: A002 - mirror torch's name
            # `dim` is accepted and ignored: sigmoid is elementwise, which is
            # precisely the point -- scores stop competing across labels.
            return torch.sigmoid(input)

        def __getattr__(self, name):
            return getattr(_F, name)

    predictor.F = _SigmoidFunctional()

    # Fail loudly rather than silently producing softmax numbers under a
    # sigmoid label: verify the patch is actually in effect.
    probe = torch.tensor([[2.0, 1.0]])
    patched = predictor.F.softmax(probe, dim=-1)
    expected = torch.sigmoid(probe)
    if not torch.allclose(patched, expected):
        raise RuntimeError(
            "Sigmoid patch failed to take effect; refusing to run so that the "
            "ablation cannot be silently mislabelled."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run fold inference with sigmoid instead of softmax confidence "
            "scores. All other arguments are passed through unchanged to "
            "cardioner.run_inference_on_folds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Any additional arguments are forwarded to run_inference_on_folds.",
    )
    parser.add_argument(
        "--activation",
        choices=["softmax", "sigmoid"],
        default="sigmoid",
        help=(
            "Confidence scale used for the decoding thresholds. "
            "'softmax' reproduces the original pipeline exactly (control run); "
            "'sigmoid' is the ablation."
        ),
    )
    args, passthrough = parser.parse_known_args()

    if args.activation == "sigmoid":
        patch_predictor_to_sigmoid()

    banner = f"  DECODING ABLATION: confidence scale = {args.activation.upper()}  "
    print("\n" + "=" * len(banner))
    print(banner)
    print("=" * len(banner))
    if args.activation == "sigmoid":
        print(
            "  Label selection is unchanged (argmax over raw logits).\n"
            "  Only threshold confidences differ."
        )
    else:
        print("  Control run: identical to the reported pipeline.")
    print("=" * len(banner) + "\n")

    from cardioner import run_inference_on_folds

    # run_main() parses sys.argv itself; hand it everything we did not consume.
    sys.argv = [sys.argv[0]] + passthrough
    run_inference_on_folds.run_main()


if __name__ == "__main__":
    main()
