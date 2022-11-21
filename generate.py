import argparse
import torch
import yaml
from utils import save_image
from networks.stylegan import Generator

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    parser = argparse.ArgumentParser(description="Train network")
    parser.add_argument("--config", type=str, help="config path")
    parser.add_argument("--out", type=str, help="generated image name")
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--truncation", type=float, help="truncation value", default=0.6)

    args = parser.parse_args()
    
    with open(args.config, "r") as data:
        config = yaml.safe_load(data)

    generator = Generator(config["resolution"], config["z_dim"], config["w_dim"], device, config).to(device)
    checkpoint = torch.load(args.model)
    generator.load_state_dict(checkpoint["g_ema_state_dict"])

    noise = torch.randn(36, config["z_dim"], device=device)
    w_mean = generator.w_mean()

    fixed_fakes = generator(noise, 0, 64, False, truncation=args.truncation, truncation_w=w_mean)
    save_image(fixed_fakes.detach(), 36, args.out)

if __name__ == "__main__":
    main()
