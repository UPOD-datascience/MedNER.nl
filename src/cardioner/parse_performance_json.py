"""
Script to parse directory with performance metrics stored in json's. Basically turn [{'k': v}, ..] in {'k': [v,..]}
"""

import argparse
import json
import math
import os
import statistics as stat
from collections import defaultdict
from typing import Dict, List


def parse_dir(json_dir: str = None) -> Dict[str, List]:
    # Collect JSON files recursively.
    # We specifically look for 'eval_results.json' which is the standard output name from Trainer.
    res_dict = defaultdict(list)

    if not os.path.exists(json_dir):
        print(f"Directory {json_dir} does not exist.")
        return res_dict

    def _collect_metrics_from_file(floc: str) -> None:
        try:
            with open(floc, "r", encoding="utf-8") as f:
                d = json.load(f)

            for k, v in d.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        res_dict[f"{k}_{_k}"].append(_v)
                else:
                    res_dict[k].append(v)
        except Exception as e:
            print(f"Error reading {floc}: {e}")

    found_eval_results = False
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file == "eval_results.json":
                found_eval_results = True
                _collect_metrics_from_file(os.path.join(root, file))

    # Fallback: only if no eval_results.json files were found at all.
    if not res_dict and not found_eval_results:
        for root, dirs, files in os.walk(json_dir):
            for file in files:
                if file == "all_results.json":
                    _collect_metrics_from_file(os.path.join(root, file))

    return res_dict


def get_aggregates(res_dict: dict = None) -> Dict[str, Dict]:
    out_string = "Aggregate results \n"
    out_string += "=" * 50 + "\n"
    out_string += "class\tmean\tmedian\tstdev\n"

    # print(res_dict)
    agg_dict = defaultdict(dict)
    for k, v in res_dict.items():
        # print(v)

        # Ensure we are working with numbers
        numeric_v = [x for x in v if isinstance(x, (int, float))]

        if not numeric_v:
            continue

        _mean = sum(numeric_v) / len(numeric_v)
        _median = stat.median(numeric_v)

        if len(numeric_v) > 1:
            _stdev = stat.stdev(numeric_v)
        else:
            _stdev = 0.0

        out_string += (
            f"{k}\t{round(_mean, 3)}\t{round(_median, 3)}\t{round(_stdev, 3)}\n"
        )
        out_string += "-" * 50 + "\n"

        agg_dict[k] = {
            "mean": round(_mean, 3),
            "median": round(_median, 3),
            "stdev": round(_stdev, 3),
        }
    print(out_string)
    return agg_dict


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", type=str, required=True)

    args = argparser.parse_args()

    collected_dict = parse_dir(args.dir)

    # Save collected results
    output_collected = os.path.join(args.dir, "collected_results.json")
    with open(output_collected, "w", encoding="utf-8") as f:
        json.dump(collected_dict, f, indent=4)

    aggregated_dict = get_aggregates(collected_dict)

    # Save aggregated results
    output_aggregated = os.path.join(args.dir, "aggregated_results.json")
    with open(output_aggregated, "w", encoding="utf-8") as f:
        json.dump(aggregated_dict, f, indent=4)
