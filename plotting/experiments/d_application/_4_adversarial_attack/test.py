import wandb

api = wandb.Api() # .artifact()

#artifact = api.artifact("ga92xug/SL-Application/artifact:LinfDeepFoolAttack")
#print(artifact)
run = api.run("ga92xug/SL-Application/xe6hgfkd")

tmp = run.summary["L2DeepFoolAttack"]
robust_accs = tmp[0]
epsilons = tmp[1]
print(robust_accs, type(robust_accs))
print(epsilons)

#tmp = run.use_artifact("LinfDeepFoolAttack:latest")
#print(tmp)

#table_names = [artifact.name for artifact in run.use_artifact()]
#print("Available tables in the run:", table_names)