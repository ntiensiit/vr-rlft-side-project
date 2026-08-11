import json

from grasping_ai.pipelines.evaluate import write_evaluation_report


def test_evaluation_tracking(tmp_path):
    report_path = tmp_path / "report.json"
    log_dir = tmp_path / "tb_eval_logs"

    results = {
        "success_rate": 0.8,
        "collision_free_rate": 0.9,
        "force_closure_rate": 0.85,
    }

    write_evaluation_report(report_path, results, experiment_log_dir=log_dir)

    assert report_path.is_file()
    with report_path.open("r") as fp:
        loaded = json.load(fp)
    assert loaded == results

    assert log_dir.is_dir()
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0
