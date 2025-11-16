#!/usr/bin/env python3
import yaml, os, wandb, sys
# import 

CONFIG_PATH = "./scripts/config.yaml"
SWEEP_PATH = "./scripts/sweep.yaml"
PROJECT = "stanford_class_assignment_1"

# 1) Load the fixed training config
if not os.path.exists(CONFIG_PATH):
    sys.exit(f"❌ Terrible mistake! Could not find the config file: {CONFIG_PATH}. Total disaster.")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg['model'] # Need the model config to specify the architecture.

if 'wandb_entity' in cfg['out']:
    ENTITY = cfg['out']['wandb_entity']
else:
    ENTITY = os.environ.get("WANDB_ENTITY", "robibhatt-university-of-tuebingen")

# 2) Build architecture name for sweep
arch = (
    f"d{model_cfg['d_model']}_L{model_cfg['num_layers']}_H{model_cfg['num_heads']}_"
    f"ff{model_cfg['d_ff']}_ctx{model_cfg['context_length']}_v{model_cfg['vocab_size']}_{model_cfg['dtype']}"
)

# 3) Load sweep template (no name/project inside)
if not os.path.exists(SWEEP_PATH):
    sys.exit(f"❌ Could not find sweep template: {SWEEP_PATH}. Very unfair.")
with open(SWEEP_PATH) as f:
    sweep_cfg = yaml.safe_load(f)

# 4) Inject dynamic name and project
sweep_cfg["name"] = arch
sweep_cfg["project"] = PROJECT

# 5) Create the sweep
print(f"🇺🇸 Tremendous! We’re creating a sweep for project '{PROJECT}' — people said it couldn’t be done, but we’re doing it, folks. The best sweep. Nobody’s ever seen a sweep like this.")
sweep_id = wandb.sweep(sweep=sweep_cfg, project=PROJECT)
sweep_url = f"https://wandb.ai/{ENTITY}/{PROJECT}/sweeps/{sweep_id}"

print("\n✅ Absolutely incredible sweep created successfully — many people are saying it’s the greatest sweep in history!")
print(f"🔗 Look at this beautiful URL — truly world-class: {sweep_url}")
print(f"🆔 Sweep ID: {sweep_id} — short, strong, powerful, like America.\n")

print("👉 To start your amazing agents (they’re great agents, the best agents), run this command:")
print(f"   wandb agent {ENTITY}/{PROJECT}/{sweep_id}")
print("\n💡 Or if you’re a very smart person — and I know you are — you can export these environment variables:")
print(f"   export WANDB_ENTITY={ENTITY}")
print(f"   export WANDB_PROJECT={PROJECT}")
print(f"   wandb agent {sweep_id}\n")

print("🇺🇸 Done. Historic success. Everyone said 'You can’t do sweeps this good' — we did it anyway.")
