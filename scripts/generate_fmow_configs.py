"""Generate matching FMoW ablations without reading the broken output files."""
import argparse
import copy
from datetime import datetime
from pathlib import Path
import shutil

import yaml


def generate(rounds=200, environment="fedscale_open", output_dir=None):
    root = Path(__file__).resolve().parents[1]
    template = root / "benchmark/configs/fmow/fmow_temporal_epsilon_200.yml"
    base = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(base.get("job_conf"), list):
        raise ValueError("Template must contain a job_conf list")
    common = {}
    for entry in base["job_conf"]:
        if not isinstance(entry, dict):
            raise ValueError("Each job_conf entry must be a mapping")
        common.update(entry)
    flags = ("model_impact_aware_reclustering", "model_impact_smoothing",
             "temporal_epsilon_reclustering")
    for flag in flags:
        common.pop(flag, None)
    common.update(rounds=rounds, num_participants=10, local_steps=10,
                  eval_interval=1, test_bsz=64, model_impact_threshold=0.05,
                  model_impact_smoothing_window=3, temporal_epsilon_window=50,
                  epsilon_percentile=95, persistence_window=5,
                  max_impact_deferrals=10)
    destination = Path(output_dir) if output_dir else template.parent
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for variant, enabled in (("baseline", ()), ("impact", flags[:1]),
                             ("smoothing", flags[:2]), ("temporal", flags)):
        name = "fmow_{}_{}".format(variant, rounds)
        config = copy.deepcopy(base)
        config["worker_ips"] = ["localhost:[4,0]"]
        config["setup_commands"] = [
            "source $HOME/miniconda3/bin/activate " + environment]
        params = dict(common, job_name=name)
        # Omit disabled flags: argparse type=bool treats the string False as true.
        params.update({flag: True for flag in enabled})
        config["job_conf"] = [{key: value} for key, value in params.items()]
        text = yaml.safe_dump(config, sort_keys=False)
        if yaml.safe_load(text) != config:
            raise ValueError("Generated configuration failed YAML validation")
        path = destination / (name + ".yml")
        if path.exists():
            backup = path.with_name(path.name + ".bak_" + stamp)
            shutil.copy2(path, backup)
            print("Backed up:", backup)
        path.write_text(text, encoding="utf-8")
        if yaml.safe_load(path.read_text(encoding="utf-8")) != config:
            raise ValueError("Written configuration failed validation: " + str(path))
        print("Validated:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--environment", default="fedscale_open")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if not args.environment or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.environment):
        parser.error("--environment must be a simple conda environment name")
    generate(args.rounds, args.environment, args.output_dir)
