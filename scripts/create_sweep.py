import yaml, os
import wandb

# 1) Load your fixed config (train YAML)
with open("scripts/config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2) Build an architecture name from fixed hypers
arch = f"d{cfg['d_model']}_L{cfg['num_layers']}_H{cfg['num_heads']}_ff{cfg['d_ff']}_ctx{cfg['context_length']}_v{cfg['vocab_size']}_{cfg['dtype']}"

# 3) Load the sweep template (no name/project inside; just method/metric/command/parameters)
with open("scripts/sweep.yaml") as f:
    sweep_cfg = yaml.safe_load(f)

# 4) Inject dynamic name (and project if you want)
sweep_cfg["name"] = arch
sweep_cfg["project"] = "stanford_class_assignment_1"

# 5) Create the sweep, print SWEEP_ID
sweep_id = wandb.sweep(sweep=sweep_cfg, project=sweep_cfg["project"])
print(sweep_id)