import torch
import yaml
from networks.stylegan import Generator

CONFIG_PATH = "configs/stylegan_anime_64.yaml"
MODEL_PATH_FROM = "checkp7.pt"
MODEL_PATH_TO = "models/gen64.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(CONFIG_PATH, "r") as data:
    config = yaml.safe_load(data)

generator = Generator(config["resolution"], config["z_dim"], config["w_dim"], device, config).to(device)
checkpoint = torch.load(MODEL_PATH_FROM, map_location=torch.device(device))
generator.load_state_dict(checkpoint["g_ema_state_dict"])
checkpoint = {
    "g_ema_state_dict": generator.state_dict(),
                
}
torch.save(checkpoint, MODEL_PATH_TO)
