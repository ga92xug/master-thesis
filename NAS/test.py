import sys
#sys.path.append('')
try:
    from NAS.runner import HydraWandbRunner
except ModuleNotFoundError:
    from NAS.runner import HydraWandbRunner


class DummyTrial:
    def __init__(self, index, parameters):
        self.index = index
        self.parameters = parameters

def main():
    trial_index = 0
    trial_parameters = {
        "model.depth": 22,
        "wandb.mode": "disabled",
        "wandb.group": "NAS_test"
    }
    trial = DummyTrial(trial_index, trial_parameters)
    project_name = "scaling-laws-eq"
    script_path = "experiment/main.py"  
    runner = HydraWandbRunner(script_path, project_name)
    runner.run(trial)


if __name__ == "__main__":
    main()