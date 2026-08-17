#!/usr/bin/env pwsh
# scripts/run_full_scripts.ps1
# Run the full grasping_ai pipeline by invoking each script under scripts/
# in the canonical order, instead of going through the subprocess driver in
# scripts/run_artifacts.py. Use this when you want one terminal per stage or
# want to swap a single stage for debugging.
#
# Usage:
#   pwsh scripts/run_full_scripts.ps1                 # run every stage
#   pwsh scripts/run_full_scripts.ps1 -DryRun         # print the plan, run nothing
#   pwsh scripts/run_full_scripts.ps1 -SkipRL         # skip the RL training stage
#   pwsh scripts/run_full_scripts.ps1 -SkipSimEval    # skip simulation + evaluation
#   pwsh scripts/run_full_scripts.ps1 -Visualize      # add the interactive MuJoCo viewer stage
#   pwsh scripts/run_full_scripts.ps1 -Help
#   pwsh scripts/run_full_scripts.ps1 seed=7          # forward a Hydra override

[CmdletBinding()]
param(
    [switch]$Help,
    [switch]$DryRun,
    [switch]$SkipRL,
    [switch]$SkipSimEval,
    [switch]$ContinueOnError,
    [switch]$Visualize,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$HydraOverrides
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $here "..")

function Show-Help {
    @"
scripts/run_full_scripts.ps1 — run the full pipeline script-by-script.

PARAMETERS
    -Help                    Show this help and exit.
    -DryRun                  Print every command that would run, then exit.
    -SkipRL                  Skip the train_rl stage (the slowest one).
    -SkipSimEval             Skip run_simulation and evaluate (fast iteration on training).
    -ContinueOnError         Keep going if a stage fails. Default is stop-on-error,
                             matching the behavior of scripts/run_artifacts.py.
    -Visualize               Add the interactive MuJoCo viewer stage (scripts/visualize_robot.py).
                             Off by default — the viewer blocks until the window is closed,
                             so it is excluded from the non-interactive pipeline.

REMAINING ARGUMENTS
    Forwarded as Hydra overrides to every stage. Useful for ``seed=7``,
    ``supervised.num_epochs=10``, ``script.object_id=003_cracker_box``, etc.

    When -Visualize is set, the viewer stage also accepts:
      --vis-object-id=<id>     Object to load in the viewer.
                               Default: 003_cracker_box
                               (matches configs/object/default.yaml objects.ids.0).
      --vis-grasp-file=<path>  Optional .npy of grasp poses (K, 4, 4) to apply.
                               Default: empty (open viewer without applying a grasp).
                               Tip: pass artifacts/exports/diffusion_grasp_poses_<id>.npy.
    Every key listed below under each stage is a valid Hydra override.

PIPELINE STAGES (every Hydra override each stage consumes)

    1. prepare_ycb_mjcf  — scripts/prepare_ycb_mjcf.py
         Writes: data/processed/ycb_mjcf/<id>/object.xml (per YCB object).
         Overrides:
           script.ycb_root     data/raw/ycb            # input YCB meshes
           script.ycb_mjcf     data/processed/ycb_mjcf  # MJCF output root

    2. prepare_data  — scripts/prepare_data.py
         Writes: data/processed/<id>.npz + data/processed/index.json.
         Modes:
           prepare.mode=synthetic  (default — generate grasps, slow)
           prepare.mode=index      (rebuild index.json only, fast)
         Overrides — dataset I/O:
           script.mode            prepare.mode       # synthetic | index
           script.output_dir      prepare.output_dir # overrides dataset_root
           script.quality_report  prepare.quality_report
           script.dataset_root    paths.dataset_root # data/processed
           script.object_ids      objects.ids        # list of YCB ids
           script.robot_xml       robot.description  # deploy/robot.xml
           script.ycb_root        paths.ycb_root     # data/raw/ycb
         Overrides — synthetic data (configs/data/default.yaml):
           script.allow_relaxed               synthetic.allow_relaxed
           script.candidate_multiplier        synthetic.candidate_multiplier
           script.collision_clearance         synthetic.collision_clearance
           script.friction_coefficient        synthetic.friction_coefficient
           script.gripper_width               synthetic.gripper_width
           script.lift_height_threshold       synthetic.lift_height_threshold
           script.max_angular_velocity        synthetic.max_angular_velocity
           script.max_linear_velocity         synthetic.max_linear_velocity
           script.min_grasp_rotation          synthetic.min_grasp_rotation
           script.min_grasp_translation       synthetic.min_grasp_translation
           script.min_quality_score           synthetic.min_quality_score
           script.neighborhood_size           synthetic.neighborhood_size
           script.num_grasps                  synthetic.num_grasps
           script.num_samples                 synthetic.num_samples
           script.num_simulation_steps        synthetic.num_simulation_steps
           script.oversample_extra            synthetic.oversample_extra
           script.oversample_factor           synthetic.oversample_factor
           script.relaxed_antipodal_dot       synthetic.relaxed_antipodal_dot
           script.search_multiplier           synthetic.search_multiplier
           script.seed                        synthetic.seed
           script.sim_validate                synthetic.sim_validate
           script.sim_validate_fallback_analytical  synthetic.sim_validate_fallback_analytical
           script.sim_validate_min_contacts   synthetic.sim_validate_min_contacts
           script.sim_validate_require_ik     synthetic.sim_validate_require_ik
           script.sim_validate_require_lift   synthetic.sim_validate_require_lift
           script.strict_alignment_dot        synthetic.strict_alignment_dot
           script.strict_antipodal_dot        synthetic.strict_antipodal_dot
           script.voxel_size                  synthetic.voxel_size

    3. prepare_observations  — scripts/prepare_observations.py
         Writes: data/observations/<id>.npy + merged_objects*.npy + gripper.npy.
         Overrides:
           script.ycb_root         paths.ycb_root      # data/raw/ycb
           script.observations_dir paths.observations  # data/observations
           script.num_samples      observations.num_samples
           script.seed             observations.seed

    4. train_diffusion  — scripts/train_diffusion.py
         Writes: artifacts/checkpoints/diffusion_grasp_generator.pt.
         Overrides — supervised training (configs/training/supervised.yaml):
           supervised.batch_size       # default 2
           supervised.learning_rate    # default 0.001
           supervised.num_epochs       # default 3
           supervised.min_grasp_score  # default 0.0
           supervised.score_repeat_factor
           supervised.score_repeat_power
         Overrides — training loop (configs/training/supervised.yaml):
           training.augment             # default false
           training.resume              # path to checkpoint
           training.pretrained_encoder  # path to encoder ckpt
           training.log_every
         Overrides — model + run:
           model.checkpoint             # default artifacts/checkpoints/diffusion_grasp_generator.pt
           diffusion.tensorboard        # default artifacts/exports/tensorboard/diffusion_train
           device                       # default cpu
           seed                         # default 42
           paths.dataset_root           # default data/processed
           architecture.feature_dim     # default 32
           architecture.hidden_dim      # default 32
           architecture.num_layers      # default 2

    5. generate_grasps  — scripts/generate_grasps.py
         Writes: artifacts/exports/diffusion_grasp_candidates_by_object.npy.
         Overrides:
           script.checkpoint     model.checkpoint
           script.observations   observations.files  # list of per-object .npy paths
           script.output         model.exports.grasp_candidates
           script.feature_dim    architecture.feature_dim
           script.num_grasps     architecture.num_grasps
           script.device         device
           script.seed           seed

    6. extract_object_grasps  — scripts/extract_object_grasps.py
         Writes: artifacts/exports/diffusion_grasp_poses_<objects.ids.0>.npy.
         Overrides:
           script.input    model.exports.grasp_candidates
           script.output   model.exports.grasp_poses
           script.key      evaluation.single_object_key  # default object_0

    7. run_simulation  — scripts/run_simulation.py
         Writes: artifacts/reports/diffusion_simulation_outcomes_<id>.jsonl.
         Overrides:
           script.grasps             model.exports.grasp_poses
           script.object_ids         objects.ids
           script.object_id          objects.ids.0       # default 003_cracker_box
           script.output             model.exports.simulation_report
           script.grasp_pose_format  world              # default; world | object
           script.ycb_root           paths.ycb_mjcf      # data/processed/ycb_mjcf
           script.robot_xml          robot.description   # deploy/robot.xml
           script.table_xml          table_xml           # deploy/table.xml
           script.num_steps          num_steps           # default 50

    8. evaluate  — scripts/evaluate.py
         Writes: artifacts/reports/diffusion_analytical_evaluation_report.jsonl.
         Overrides:
           script.object_id              evaluation.single_object_key  # default object_0
           script.multi_object           # default false (run_artifacts sets true)
           script.grasps                 model.exports.grasp_candidates
           script.report                 evaluation.analytical_report
           script.filter_collisions      evaluation.filter_collisions
           script.observations_dir       paths.observations
           script.object_ids             objects.ids
           script.gripper_point_cloud    observations.gripper_point_cloud
           script.friction_coefficient   metrics.friction_coefficient
           script.lift_height_threshold  metrics.lift_height_threshold
           script.contact_clearance      metrics.collision_clearance
           script.wrench_regularization  metrics.wrench_regularization

    9. train_rl  — scripts/train_rl.py
         Writes: artifacts/checkpoints/rl_grasp_policy.pt.
         Overrides:
           script.robot_xml            robot.description
           script.ycb_root             paths.ycb_mjcf
           script.object_ids           objects.ids
           script.policy_checkpoint    rl.checkpoint
           script.observation_dim      rl.observation_dim   # default 31
           script.action_dim           rl.action_dim        # default 8
           script.hidden_dim           rl.hidden_dim        # default 32
           script.learning_rate        rl.learning_rate     # default 0.0003
           script.num_updates          rl.num_updates       # default 10
           script.gamma                rl.gamma             # default 0.99
           script.n_steps              rl.n_steps           # default 64
           script.batch_size           rl.batch_size        # default 64
           script.n_epochs             rl.n_epochs          # default 1
           script.policy_num_layers    rl.policy_num_layers # default 2
           script.device               device
           script.seed                 seed

   10. visualize_robot  — scripts/visualize_robot.py  [only when -Visualize is set]
         Writes: nothing — opens an interactive MuJoCo viewer for the robot scene
                 (and optionally applies one grasp pose). Blocks until the window closes.
         Overrides:
           script.object_id              null             # YCB object id; null = none
           script.grasp_file             null             # .npy (K, 4, 4); null = no grasp
           script.grasp_index            0                # index into grasp_file
           script.allow_ik_failure       true             # if IK fails, show pose anyway
           script.auto_select_reachable  true             # scan all grasps for a reachable one
           script.close_gripper          true             # close the gripper in the viewer
           script.grasp_pose_format      object           # object | world
           script.poses_ndim             grasp.poses_ndim  # default 3
           script.robot_xml              robot.description
           script.ycb_root               paths.ycb_mjcf
           script.table_xml              table_xml
         Hydra overrides (extracted from remaining args):
           --vis-object-id=<id>     sets script.object_id (default 003_cracker_box)
           --vis-grasp-file=<path>  sets script.grasp_file (default empty)

ENVIRONMENT
    Sets PYTHONPATH=<root>/src and PYTHONPYCACHEPREFIX=<root>/.pycache so the
    grasping_ai package is importable and bytecode stays out of the source tree.

EXAMPLES
    pwsh scripts/run_full_scripts.ps1
    pwsh scripts/run_full_scripts.ps1 -DryRun
    pwsh scripts/run_full_scripts.ps1 -SkipRL seed=42
    pwsh scripts/run_full_scripts.ps1 -SkipSimEval supervised.num_epochs=5000 supervised.batch_size=8

    Add the interactive viewer at the end of the pipeline:
    pwsh scripts/run_full_scripts.ps1 -Visualize
    pwsh scripts/run_full_scripts.ps1 -Visualize -SkipRL --vis-object-id=004_sugar_box

    Forward stage-specific overrides to every stage:
    pwsh scripts/run_full_scripts.ps1 seed=42
    pwsh scripts/run_full_scripts.ps1 supervised.num_epochs=5000 supervised.batch_size=8 rl.num_updates=500
    pwsh scripts/run_full_scripts.ps1 synthetic.num_samples=2048 synthetic.num_grasps=64
    pwsh scripts/run_full_scripts.ps1 observations.num_samples=512 seed=42
"@
}

if ($Help) {
    Show-Help
    exit 0
}


# Extract --vis-object-id= and --vis-grasp-file= from overrides when -Visualize is set.
$visObjectId = "003_cracker_box"
$visGraspFile = ""
$filteredOverrides = [System.Collections.Generic.List[string]]::new()
foreach ($o in $HydraOverrides) {
    if ($o -match '^--vis-object-id=(.+)$') {
        $visObjectId = $Matches[1]
    } elseif ($o -match '^--vis-grasp-file=(.+)$') {
        $visGraspFile = $Matches[1]
    } else {
        $filteredOverrides.Add($o)
    }
}
$HydraOverrides = $filteredOverrides.ToArray()

# Build the ordered stage list. Each entry is @{ Name; Script; Args }.
$stages = New-Object System.Collections.Generic.List[object]
$stages.Add([pscustomobject]@{ Name = "prepare_ycb_mjcf"; Script = "prepare_ycb_mjcf"; Args = @() })
$stages.Add([pscustomobject]@{ Name = "prepare_data (synthetic)"; Script = "prepare_data"; Args = @("prepare.mode=synthetic") })
$stages.Add([pscustomobject]@{ Name = "prepare_observations"; Script = "prepare_observations"; Args = @() })
$stages.Add([pscustomobject]@{ Name = "train_diffusion"; Script = "train_diffusion"; Args = @() })
$stages.Add([pscustomobject]@{ Name = "generate_grasps"; Script = "generate_grasps"; Args = @() })
$stages.Add([pscustomobject]@{ Name = "extract_object_grasps"; Script = "extract_object_grasps"; Args = @() })
if ($Visualize) {
    $visArgs = New-Object System.Collections.Generic.List[string]
    $visArgs.Add("script.object_id=$visObjectId")
    if ($visGraspFile -ne "") {
        $visArgs.Add("script.grasp_file=$visGraspFile")
    }
    $stages.Add([pscustomobject]@{ Name = "visualize_robot"; Script = "visualize_robot"; Args = $visArgs.ToArray() })
}
if (-not $SkipSimEval) {
    $stages.Add([pscustomobject]@{ Name = "run_simulation"; Script = "run_simulation"; Args = @() })
    $stages.Add([pscustomobject]@{ Name = "evaluate"; Script = "evaluate"; Args = @("script.multi_object=true") })
}
if (-not $SkipRL) {
    $stages.Add([pscustomobject]@{ Name = "train_rl"; Script = "train_rl"; Args = @() })
}

$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONPYCACHEPREFIX = Join-Path $root ".pycache"

Push-Location -LiteralPath $root
try {
    $total = $stages.Count
    $index = 0
    foreach ($stage in $stages) {
        $index++
        $scriptPath = Join-Path $root "scripts" "$($stage.Script).py"
        if (-not (Test-Path -LiteralPath $scriptPath)) {
            Write-Error "Script not found: $scriptPath"
            if (-not $ContinueOnError) { exit 2 }
            continue
        }
        $argv = @($scriptPath) + $stage.Args + $HydraOverrides
        $cmdline = "uv run --no-sync python $($argv -join ' ')"
        Write-Host ""
        Write-Host ("[{0}/{1}] {2}" -f $index, $total, $stage.Name)
        Write-Host "  $ $cmdline"
        if ($DryRun) { continue }
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        uv run --no-sync python @argv
        $sw.Stop()
        $code = $LASTEXITCODE
        Write-Host ("  -> exit {0} in {1:n1}s" -f $code, $sw.Elapsed.TotalSeconds)
        if ($code -ne 0) {
            if ($ContinueOnError) {
                Write-Warning "Stage '$($stage.Name)' failed with exit $code; continuing."
            } else {
                Write-Error "Stage '$($stage.Name)' failed with exit $code."
                exit $code
            }
        }
    }
    Write-Host ""
    if ($DryRun) {
        Write-Host "Dry run complete; no commands were executed."
    } else {
        Write-Host "All stages completed."
    }
}
finally {
    Pop-Location
}
