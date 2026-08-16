# Phase 7: Synthetic Data Generation and Ground Truth Grasps

> **Historical design record.** This document captures the Phase 7 plan from the skeleton era. Current architecture, CLIs, and contracts live in [architecture.md](architecture.md), [USAGE.md](USAGE.md), and the repository [README.md](../README.md).

## 1. Phase Overview
**Name:** Phase 7: Synthetic Data Generation and Ground Truth Grasps  
**Objective:** Implement deterministic mesh-to-point-cloud sampling and analytical grasp generation to produce valid, paired training datasets from YCB 3D assets.  
**Why this phase is necessary:** The generative models implemented in Phase 4 require paired data consisting of point clouds and SE(3) grasp poses. The repository currently includes YCB 3D meshes, but the function `sample_point_cloud_from_mesh` in `src/grasping_ai/sensors/pointcloud_sensor.py` is an explicit stub. Furthermore, no analytical grasp sampler exists in the codebase to generate ground truth grasps from these meshes. Without this capability, the supervised training pipeline cannot be executed locally, rendering the generative models untrainable.  
**Expected outcome:** A functional, fully local data generation pipeline that consumes YCB meshes and outputs `.npy` records strictly compatible with the existing `iterate_grasp_dataset` contract.

## 2. Verified Current State
*   **`src/grasping_ai/sensors/pointcloud_sensor.py`**: Contains the stub `sample_point_cloud_from_mesh` which explicitly raises `NotImplementedError`. The stubs `acquire_point_cloud_stream` and `merge_point_clouds` also exist but relate to real-sensor streaming and are out of scope.
*   **`src/grasping_ai/data/pointcloud_dataset.py`**: Contains `discover_dataset_files`, `load_grasp_sample`, `iterate_grasp_dataset`, and `resolve_ycb_object_id`. It defines the exact data contract expected by downstream phases. Missing analytical grasp generation logic.
*   **`scripts/prepare_data.py`**: Currently only discovers existing `.npy` files in a directory and writes an index via `save_grasp_dataset_index`. It does not generate synthetic data.
*   **`src/grasping_ai/perception/pointcloud.py`**: Contains `estimate_point_cloud_normals` which can be reused for normal estimation on sampled points.
*   **`pyproject.toml`**: Confirms `open3d>=0.19.0`, `numpy>=1.26.0`, and `scipy>=1.18.0` are available as dependencies.
*   **`src/grasping_ai/evaluation/collision.py`**: Contains `build_collision_checker` using KDTree, but it is designed for gripper-vs-object collision, not analytical grasp width validation.

## 3. Phase Boundary
**Exact scope:** 
1. Implement deterministic triangle-based mesh sampling in `sample_point_cloud_from_mesh`.
2. Implement analytical antipodal grasp generation in `src/grasping_ai/data/pointcloud_dataset.py`.
3. Extend `scripts/prepare_data.py` to orchestrate synthetic data generation and serialization.

**Explicitly excluded responsibilities:**
*   Implementation of `acquire_point_cloud_stream` and `merge_point_clouds` (real-sensor streaming).
*   Multi-view depth image fusion or real-sensor merging.
*   Modifications to generative model architectures, training loops, or loss functions.
*   MuJoCo simulation-based grasp validation or dynamic simulation execution.
*   Physical robot deployment or hardware communication.

**MECE boundary definition:** This phase strictly owns offline synthetic data generation from 3D meshes. It does not own data loading/iteration (Phase 3), model training (Phase 4), or evaluation (Phase 6/10).

## 4. Architecture and Dependency Analysis
*   **Input:** YCB mesh files (`.obj` or `.ply`) resolved via `src/grasping_ai/simulation/ycb.py`.
*   **Processing:** Mesh parsing -> Triangle area weighting -> Deterministic point sampling -> Normal estimation -> Antipodal grasp sampling -> SE(3) conversion.
*   **Output:** `.npy` records containing `point_cloud` (N, 3) and `grasp_poses` (K, 4, 4), serialized exactly as expected by `load_grasp_sample`.
*   **Dependencies:** `open3d` for mesh parsing, `numpy` for vectorized math, `scipy.spatial.KDTree` for neighbor queries, and `pytransform3d` (via existing perception modules) for SE(3) conventions.
*   **Simulation/RL/Eval dependencies:** None. This phase is strictly offline and geometric.

## 5. Source Code Impact Analysis

### `src/grasping_ai/sensors/pointcloud_sensor.py`
*   **Current responsibility:** Sensor observation acquisition.
*   **Exact function affected:** `sample_point_cloud_from_mesh`
*   **Current behavior:** Raises `NotImplementedError`.
*   **Required change:** Implement deterministic triangle-based mesh sampling using the provided `rng`.
*   **Reason for change:** Fulfill the stub contract and enable point cloud generation.
*   **Dependency impact:** Requires `open3d` import inside the function or at module level.
*   **Regression risk:** Low. The function was previously a stub.

### `src/grasping_ai/data/pointcloud_dataset.py`
*   **Current responsibility:** Dataset loading, discovery, and transforms.
*   **Exact function affected:** New function `generate_analytical_grasps` to be added.
*   **Current behavior:** N/A (Missing).
*   **Required change:** Add vectorized analytical grasp generation logic.
*   **Reason for change:** Provide ground truth SE(3) grasps for training.
*   **Dependency impact:** Requires `scipy.spatial.KDTree` and existing `perception.pointcloud.estimate_point_cloud_normals`.
*   **Regression risk:** Low. Additive change.

### `scripts/prepare_data.py`
*   **Current responsibility:** Thin CLI wrapper for dataset indexing.
*   **Exact function affected:** `prepare_data` and `__main__` block.
*   **Current behavior:** Only indexes existing files.
*   **Required change:** Add a `--mode synthetic` branch that orchestrates mesh loading, sampling, grasp generation, and `.npy` serialization.
*   **Reason for change:** Provide a CLI entry point for the new data generation pipeline.
*   **Dependency impact:** Imports new synthetic generation functions.
*   **Regression risk:** Medium. Must ensure the original `--mode index` (default) behavior remains completely unchanged.

## 6. New Source Code Units

### `generate_analytical_grasps` in `src/grasping_ai/data/pointcloud_dataset.py`
*   **Responsibility:** Sample antipodal point pairs, construct SE(3) grasp poses, and filter by gripper width.
*   **Inputs:** `points` (N, 3), `normals` (N, 3), `num_grasps` (int), `gripper_width` (float), `rng` (np.random.Generator).
*   **Outputs:** `grasp_poses` (K, 4, 4) numpy array.
*   **Failure behavior:** Returns an empty array (0, 4, 4) if no valid grasps are found within the attempt limit.
*   **Why existing code cannot be extended:** No existing grasp generation logic exists in the repository. This is a core data generation capability, not a utility.

## 7. Existing Code Modification Rules
*   **Preserve `load_grasp_sample` contract:** The generated `.npy` files must contain exactly the keys `point_cloud`, `grasp_poses`, `scores`, and `object_id`.
*   **Preserve `sample_point_cloud_from_mesh` signature:** Must strictly accept `(mesh_path: Path, num_samples: int, rng: np.random.Generator)`.
*   **No private helpers:** All vectorized math for triangle sampling and grasp construction must be implemented directly inside the target functions using numpy broadcasting. No private `_helper` functions are permitted.
*   **No global state:** All randomness must flow strictly through the `rng` parameter.

## 8. Detailed Behavioral Design

### `sample_point_cloud_from_mesh`
1. Load mesh using `open3d.io.read_triangle_mesh`.
2. Extract vertices and triangles as numpy arrays.
3. Compute triangle areas using vectorized cross products.
4. Sample `num_samples` triangle indices using `rng.choice` weighted by normalized triangle areas.
5. Generate barycentric coordinates using `rng.random()` and `np.sqrt()` to ensure uniform distribution on the triangle surface.
6. Compute final 3D points and return as `float32` array.

### `generate_analytical_grasps`
1. Build a `scipy.spatial.KDTree` on the input `points`.
2. For a predefined number of attempts (e.g., `num_grasps * 20`):
   a. Sample a random point index `i` using `rng`.
   b. Query the KDTree for neighbors within `gripper_width`.
   c. Filter neighbors `j` where the dot product of `normals[i]` and `-normals[j]` exceeds a threshold (e.g., 0.5) to ensure antipodal alignment.
   d. If valid, construct the grasp frame: Z-axis along `points[j] - points[i]`, X-axis from the cross product of Z and the average normal, Y-axis as Z x X.
   e. Assemble the 4x4 SE(3) matrix with the midpoint as translation.
3. Return up to `num_grasps` valid matrices.

### `scripts/prepare_data.py`
1. Add `--mode` argument (`index` or `synthetic`).
2. If `synthetic`, require `--ycb-root`, `--output-dir`, `--num-samples`, `--num-grasps`, `--gripper-width`.
3. Iterate over YCB objects, resolve mesh paths, sample points, estimate normals via `perception.pointcloud.estimate_point_cloud_normals`, generate grasps, and save `.npy` records.
4. Finally, generate the index file using the existing `save_grasp_dataset_index`.

## 9. Cross-Phase Impact
*   **Phase 3 (Data Pipeline):** The generated `.npy` files must be perfectly compatible with `load_grasp_sample`. Any deviation in dictionary keys or array shapes will break Phase 3 data loading.
*   **Phase 4 (Generative Model):** The training pipeline converts SE(3) matrices to 9D vectors. The generated rotation matrices must be strictly orthonormal with determinant +1 to prevent NaN losses during training.
*   **Regression tests:** Phase 3 data loading tests and Phase 4 training data contract tests must continue to pass.

## 10. Regression Protection
*   **Unit tests:** Verify mesh sampling determinism and bounding box containment. Verify grasp generation output shapes and rotation matrix orthogonality.
*   **Integration tests:** Execute `scripts/prepare_data.py --mode synthetic` on a dummy mesh, then load the output with `load_grasp_sample` and verify keys and shapes.
*   **Simulation tests:** N/A for this phase.
*   **Numerical validation:** Verify that all generated rotation matrices satisfy `R @ R.T == I` and `det(R) == 1` within floating-point tolerance.

## 11. New Test Suite

### `tests/unit/test_synthetic_data.py`
*   **Purpose:** Validate geometric correctness and determinism of synthetic data generation.
*   **Input strategy:** Use a simple synthetic mesh (e.g., a cube or sphere) created in-memory or via a tiny `.ply` file.
*   **Expected output:** Correct point counts, valid SE(3) matrices, deterministic outputs for identical RNG seeds.
*   **Failure condition:** Non-orthogonal rotation matrices, points outside mesh bounding box, non-deterministic outputs.
*   **Determinism requirement:** Strict adherence to the provided `np.random.Generator`.
*   **CI suitability:** Fast, no external network or heavy asset dependencies.

### `tests/integration/test_prepare_data_synthetic.py`
*   **Purpose:** Validate the end-to-end CLI pipeline.
*   **Input strategy:** Run the CLI script with a temporary directory containing a dummy mesh.
*   **Expected output:** `.npy` files and `index.json` are created and readable by `iterate_grasp_dataset`.
*   **Failure condition:** CLI crashes, missing keys in serialized data, incompatible shapes.
*   **Determinism requirement:** Seeded RNG passed to the pipeline.
*   **CI suitability:** Uses temporary directories and minimal assets.

## 12. Test Matrix
| Code Unit | Affected Phase | Test | Regression Coverage | Commit Gate |
| :--- | :--- | :--- | :--- | :--- |
| `sample_point_cloud_from_mesh` | Phase 7 | `test_sample_point_cloud_from_mesh_*` | Phase 3 data loading | Unit tests pass |
| `generate_analytical_grasps` | Phase 7 | `test_generate_analytical_grasps_*` | Phase 4 training data contract | Unit tests pass |
| `scripts/prepare_data.py` | Phase 7 | `test_prepare_data_synthetic_pipeline` | Phase 3 index generation | Integration tests pass |

## 13. Numerical and Robotics Validation
*   **SE(3) validation:** All generated grasp poses must have orthonormal rotation matrices with determinant +1.
*   **Coordinate frames:** Grasp Z-axis must align with the line connecting the contact points. X and Y axes must be orthogonal to Z and the surface normals.
*   **Determinism constraints:** All random sampling must strictly use the provided `np.random.Generator`. No implicit `np.random` calls are permitted anywhere in the implementation.

## 14. Data and Interface Contracts
*   **Input mesh:** `.obj` or `.ply` readable by `open3d`.
*   **Output `.npy`:** Dictionary with keys `point_cloud` (N, 3 float32), `grasp_poses` (K, 4, 4 float32), `scores` (None or array), `object_id` (str).
*   **Units:** Meters.
*   **Coordinate system:** Mesh local frame.

## 15. Configuration and Dependency Impact
*   **Dependencies:** `open3d` is already in `pyproject.toml`. No new dependencies required.
*   **Configuration:** `scripts/prepare_data.py` will accept new CLI arguments: `--mode`, `--num-samples`, `--num-grasps`, `--gripper-width`. No global configuration files are modified.

## 16. Reproducibility and Local Validation
*   **Local execution:** Fully local. Uses YCB meshes already present in `data/raw/ycb/`.
*   **Deterministic setup:** Guaranteed via explicit `rng` seeding in the CLI and pipeline functions.
*   **Synthetic data usage:** Generates data on-demand; no pre-computed binary assets are committed to the repository.
*   **No physical robot dependency:** Strictly offline geometric processing.

## 17. Implementation Order
1. Implement `sample_point_cloud_from_mesh` in `src/grasping_ai/sensors/pointcloud_sensor.py`.
2. Add unit tests for mesh sampling in `tests/unit/test_synthetic_data.py`.
3. Implement `generate_analytical_grasps` in `src/grasping_ai/data/pointcloud_dataset.py`.
4. Add unit tests for grasp generation in `tests/unit/test_synthetic_data.py`.
5. Update `scripts/prepare_data.py` to support `--mode synthetic`.
6. Add integration test `tests/integration/test_prepare_data_synthetic.py`.
7. Run full test suite to verify no regressions in Phase 3 or Phase 4.

## 18. Commit Gates
*   All new unit and integration tests pass.
*   Existing Phase 3 and Phase 4 tests pass (no regression in data loading or model training contracts).
*   No helper functions, utility modules, or classes created.
*   Google docstrings present on all new/modified functions.
*   `ruff` and `mypy` pass without errors.
*   No architectural violations (e.g., no global state, no new abstractions).

## 19. Definition of Done
*   `sample_point_cloud_from_mesh` is fully implemented, deterministic, and passes unit tests.
*   `generate_analytical_grasps` produces valid SE(3) matrices respecting width constraints and passes numerical validation.
*   `scripts/prepare_data.py` can generate a complete synthetic dataset from YCB meshes via the CLI.
*   The generated dataset is successfully consumed by `iterate_grasp_dataset` without raising validation errors.
*   All commit gates are satisfied.

## 20. Risks and Failure Modes
*   **Technical risk:** `open3d` mesh loading fails on certain YCB `.obj` files due to missing materials or malformed geometry. 
    *   *Mitigation:* Fallback to `.ply` if available via `resolve_ycb_object_id`, or handle `open3d` exceptions gracefully and skip corrupted meshes with a warning.
*   **Numerical risk:** Analytical grasp generation yields zero valid grasps for highly convex or thin objects due to strict antipodal thresholds. 
    *   *Mitigation:* Implement a fallback sampling strategy that relaxes the normal constraint if the attempt limit is reached, ensuring the pipeline does not crash.
*   **Integration risk:** Non-deterministic behavior from `open3d` internal sampling functions. 
    *   *Mitigation:* Strictly implement triangle sampling using `numpy` and the provided `rng`, completely bypassing `open3d`'s built-in sampling functions.
*   **Regression risk:** Modifying `scripts/prepare_data.py` breaks the existing indexing workflow. 
    *   *Mitigation:* Default `--mode` to `index` and ensure the original code path is completely untouched when `--mode synthetic` is not provided.

## 21. Out of Scope
*   `acquire_point_cloud_stream` and `merge_point_clouds` (real-sensor streaming).
*   Multi-view depth image fusion.
*   Modifications to the generative model architectures or training loops.
*   MuJoCo simulation-based grasp validation (reserved for Phase 10/6).
*   Physical robot deployment or hardware communication.
*   Creation of any utility modules or helper classes.

## 22. Design Review Checklist
*   [x] Repository verified (stubs, dependencies, and contracts confirmed).
*   [x] Phase boundary correct (strictly synthetic data generation).
*   [x] No forbidden constructs (no helpers, classes, utils, or global state).
*   [x] Tests complete (unit and integration tests defined).
*   [x] No architectural violation (follows thin CLI and pure function patterns).
*   [x] Local reproducibility confirmed (uses local meshes and explicit RNG).