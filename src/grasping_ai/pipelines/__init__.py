"""End-to-end training and evaluation pipelines."""

from __future__ import annotations

from grasping_ai.pipelines.evaluate import (
    aggregate_evaluation_results,
    evaluate_generated_grasps,
    read_jsonl_records,
    write_evaluation_report,
    write_jsonl_records,
)
from grasping_ai.pipelines.generate_grasps import (
    load_generated_grasps,
    write_generated_grasps,
    write_generated_grasps_array,
)
from grasping_ai.pipelines.prepare_synthetic_data import generate_synthetic_dataset, prepare_data_index
from grasping_ai.pipelines.simulate_grasp import run_simulation_sweep, simulate_grasp

__all__ = [
    "aggregate_evaluation_results",
    "evaluate_generated_grasps",
    "generate_synthetic_dataset",
    "load_generated_grasps",
    "prepare_data_index",
    "read_jsonl_records",
    "run_simulation_sweep",
    "simulate_grasp",
    "write_evaluation_report",
    "write_generated_grasps",
    "write_generated_grasps_array",
    "write_jsonl_records",
]
