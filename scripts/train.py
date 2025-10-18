from cs336_basics.training.train import TrainConfig, train
import yaml


def main():    
    with open("scripts/config.yaml") as f:
        data = yaml.safe_load(f)

    cfg = TrainConfig(**data)
    
    train(cfg=cfg)


if __name__=="__main__":
    main()
