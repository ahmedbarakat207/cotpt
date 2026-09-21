import json
import os


class ExperimentLogger:
    def __init__(self, log_path: str, use_wandb: bool = False, wandb_project: str = None, wandb_config: dict = None):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.log_path = log_path
        self._fh = open(log_path, "a")
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=wandb_project or "cotpt", config=wandb_config or {})
            except ImportError:
                print("wandb not installed; logging to JSONL only.")

    def log(self, step: int, **metrics):
        record = {"step": step, **metrics}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(record, step=step)

    def close(self):
        self._fh.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
