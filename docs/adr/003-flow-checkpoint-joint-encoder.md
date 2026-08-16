# ADR-0003 — Flow model checkpoint contract: jointly train encoder + flow field

## Status

Accepted (2026-08-12). Replaces the train/inference inconsistency in the
flow-matching path identified in the Phase 4 review.

## Context

The original flow training pipeline (`src/grasping_ai/pipelines/train_flow.py`)
built the equivariant encoder and the flow field as two separate modules:

* a `build_equivariant_encoder(...)` instance used by the dataloader to
  compute conditioning features;
* a `build_flow_field(...)` instance optimized by a single Adam
  optimizer over only `flow_field.parameters()`.

The result was a real architectural inconsistency:

* The encoder was *not* trained jointly with the flow field; it stayed at
  its random initialization for the entire training run.
* The optimizer covered only the flow field, so the encoder weights were
  effectively frozen during training.
* The training-time encoder was not saved in the checkpoint.
* Inference code reconstructed an encoder from architecture alone, with no
  guarantee that the encoder state matched the encoder state used at
  training time.

This created a real train/inference inconsistency: the flow field was
trained against a fixed random encoder, but the checkpoint made no record
of that encoder. A downstream user reloading the checkpoint could not
reproduce the conditioning signal used during training.

## Decision

The flow path now follows the same architectural pattern as the diffusion
path (`GraspGeneratorModel`):

* A new `FlowGeneratorModel` class in `src/grasping_ai/models/flow.py`
  owns both the encoder and the flow field as submodules.
* `run_flow_training_pipeline` constructs a single `FlowGeneratorModel`,
  builds one Adam optimizer over `model.parameters()`, and trains both
  jointly on the flow-matching objective.
* `save_training_checkpoint` saves the combined `state_dict`, which
  contains both `encoder.*` and `flow_field.*` keys.
* A new `load_flow_model_checkpoint` helper reconstructs the same
  `FlowGeneratorModel` architecture and loads the combined state dict so
  inference uses the exact encoder state that training produced.
* The flow dataloader no longer carries a separately-set encoder; the
  encoder lives on the model itself.

## Rationale

* This matches the diffusion architecture (`GraspGeneratorModel`), where
  encoder and score network are jointly trained and saved together.
* It eliminates the "frozen random encoder" failure mode in the original
  flow path.
* It makes the train/inference model contract explicit: loading a
  flow checkpoint reconstructs both the encoder and the flow field with
  the exact trained weights.
* It preserves the existing CLI (`scripts/train_flow.py`) and pipeline
  signatures.

## Consequences

* Flow checkpoints are larger because they contain both the encoder and
  the flow-field weights. This is a small constant per-checkpoint cost
  and the diffusion path already pays it.
* The optimizer covers more parameters (encoder + flow field). For the
  small configurations used by the unit tests this is harmless.
* The standalone flow-field factory was removed because it had no production
  caller; all training and inference paths use `FlowGeneratorModel`.
* Inference code that wants to load a trained flow checkpoint should use
  `load_flow_model_checkpoint(...)` rather than rebuilding the
  architecture independently.

## Follow-up review triggers

This ADR should be revisited when:

* a downstream experiment reveals that encoder+flow-field joint training
  needs different learning rates or staged training (e.g., pretrained
  encoder for the first N epochs, joint training afterwards);
* the encoder is replaced by a steerable / equivariant variant and the
  contract must be revisited.