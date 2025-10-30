from cs336_basics.training.train_config import from_yaml
from cs336_basics.training.train import train

def main():
    cfg = from_yaml("scripts/config.yaml")
    train(cfg)

if __name__ == "__main__":
    main()