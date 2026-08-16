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
from grasping_ai.pipelines.train_diffusion import run_diffusion_training_pipeline
from grasping_ai.pipelines.train_flow import (
    load_flow_model_checkpoint,
    run_flow_training_pipeline,
)
from grasping_ai.pipelines.train_rl import run_rl_training_pipeline

__all__ = [
    "aggregate_evaluation_results",
    "evaluate_generated_grasps",
    "generate_synthetic_dataset",
    "load_flow_model_checkpoint",
    "load_generated_grasps",
    "prepare_data_index",
    "read_jsonl_records",
    "run_diffusion_training_pipeline",
    "run_flow_training_pipeline",
    "run_rl_training_pipeline",
    "run_simulation_sweep",
    "simulate_grasp",
    "write_evaluation_report",
    "write_generated_grasps",
    "write_generated_grasps_array",
    "write_jsonl_records",
]
