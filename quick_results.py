import pandas as pd
import os

parquet_path = "/projects/prjs2014/patching/patching_results/llava-hf_llava-1-5-7b-hf/vqa-v2/patching_results.parquet"
output_dir = os.path.dirname(parquet_path)

from src.patching.run_patching import _print_summary

rows = pd.read_parquet(parquet_path).to_dict("records")
_print_summary(rows, output_dir)