import argparse
import os
import torch
import torch.optim as optim
import copy
from utils import resolution_to_index
from dataloader import get_loader
from networks.stylegan import Generator, Discriminator
from losses import g_nonsaturating, d_logistic, d_r1regularization, g_path_length_regularization, update_ema
from utils import save_image
from contextlib import nullcontext
import tqdm
import yaml

torch.backends.cudnn.benchmarks = True

def train(
    generator,
    discriminator,
    g_ema,
    g_optimizer,
    d_optimizer,
    g_scaler,
    d_scaler,
    precision_scope,
    current_resolution,
    dataloader,
    dataset_size,
    alpha,
    config,
    device,
    fixed_noise,
    args,
    ema_path_length,
    fade_in):
    
    for i, (real_images, _) in enumerate(tqdm.tqdm(dataloader)):
        real_images = real_images.to(device)

        noise = torch.randn(real_images.shape[0], config["z_dim"], device=device)

        # train discriminator
        with precision_scope(device):
            fake_images = generator(noise, alpha, current_resolution, fade_in)
            real_scores = discriminator(real_images, alpha, current_resolution, fade_in)
            fake_scores = discriminator(fake_images.detach(), alpha, current_resolution, fade_in)
            loss_discriminator = d_logistic(real_scores, fake_scores)

        if args.precision == "autocast" and device == "cuda":
            d_optimizer.zero_grad()
            d_scaler.scale(loss_discriminator).backward()
            d_scaler.step(d_optimizer)
            d_scaler.update()
        else:
            d_optimizer.zero_grad()
            loss_discriminator.backward()
            d_optimizer.step()
        
        if i % config["d_regularize_every"] == 0:
            with precision_scope(device):
                real_images.requires_grad = True
                real_scores = discriminator(real_images, alpha, current_resolution, fade_in)
                loss_discriminator = d_r1regularization(real_images, real_scores) * config["d_regularize_every"]
            if args.precision == "autocast" and device == "cuda":
                d_optimizer.zero_grad()
                d_scaler.scale(loss_discriminator).backward()
                d_scaler.step(d_optimizer)
                d_scaler.update()
            else:
                d_optimizer.zero_grad()
                loss_discriminator.backward()
                d_optimizer.step()

        # train generator
        with precision_scope(device):
            fake_scores = discriminator(fake_images, alpha, current_resolution, fade_in)
            loss_generator = g_nonsaturating(fake_scores)

        if args.precision == "autocast" and device == "cuda":
            g_optimizer.zero_grad()
            g_scaler.scale(loss_generator).backward()
            g_scaler.step(g_optimizer)
            g_scaler.update()
        else:
            g_optimizer.zero_grad()
            loss_generator.backward()
            g_optimizer.step()
        
        if i % config["g_regularize_every"] == 0:
            with precision_scope(device):
                noise = torch.randn(int(real_images.shape[0] / 2), config["z_dim"], device=device)
                fake_images, w = generator(noise, alpha, current_resolution, fade_in, return_w=True)
                loss_generator, ema_path_length = g_path_length_regularization(fake_images, w, ema_path_length)
                loss_generator = loss_generator * config["g_regularize_every"] * 2
            if args.precision == "autocast" and device == "cuda":
                g_optimizer.zero_grad()
                g_scaler.scale(loss_generator).backward()
                g_scaler.step(g_optimizer)
                g_scaler.update()
            else:
                g_optimizer.zero_grad()
                loss_generator.backward()
                g_optimizer.step()


        # update alpha
        alpha += real_images.shape[0] / (config["n_epochs"] * dataset_size)

        update_ema(generator, g_ema)
        if i % 100 == 0:
            fixed_fakes = g_ema(fixed_noise, alpha, current_resolution, fade_in)
            save_image(fixed_fakes, 36, (args.out + "/fixed_fakes.png"))
            
        
    return alpha, ema_path_length



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    parser = argparse.ArgumentParser(description="Train network")
    parser.add_argument("--config", type=str, help="config path", required=True)
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument("--out", type=str, help="experiment path", required=True)
    parser.add_argument("--precision", type=str, help="precision", choices=["full", "autocast"], default="autocast")

    args = parser.parse_args()

    with open(args.config, "r") as data:
        config = yaml.safe_load(data)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    generator = Generator(config["resolution"], config["z_dim"], config["w_dim"], device, config).to(device)
    discriminator = Discriminator(config["resolution"], config).to(device)
    g_ema = copy.deepcopy(generator)
    d_c = config["d_regularize_every"] / (config["d_regularize_every"] + 1)
    g_c = config["g_regularize_every"] / (config["g_regularize_every"] + 1)
    g_optimizer = optim.Adam(generator.parameters(), lr=config["learning_rate"] * g_c, betas=(0.0 ** g_c, 0.99 ** g_c))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config["learning_rate"] * d_c, betas=(0.0 ** d_c, 0.99 ** d_c))

    precision_scope = torch.autocast if args.precision == "autocast" else nullcontext
    if args.precision == "autocast" and device == "cuda":
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()
    else:
        g_scaler = None
        d_scaler = None

    fixed_noise = torch.randn(36, config["z_dim"], device=device)
    current_resolution = 4
    ema_path_length = 0
    for epoch in range(resolution_to_index(config["resolution"])):
        alpha = 0
        dataloader, dataset_size = get_loader(current_resolution, config["batch_sizes"][epoch], 6, args.dataset)
        print(f"current resolution: {current_resolution}")

        for epoch in range(config["n_epochs"]):
            print("fade in")
            print(f"epoch {epoch+1}/{config['n_epochs']}")
            alpha, ema_path_length = train(
                generator,
                discriminator,
                g_ema,
                g_optimizer,
                d_optimizer,
                g_scaler,
                d_scaler,
                precision_scope,
                current_resolution,
                dataloader,
                dataset_size,
                alpha,
                config,
                device,
                fixed_noise,
                args,
                ema_path_length,
                fade_in=True)
            checkpoint = {
                "g_state_dict": generator.state_dict(),
                "g_ema_state_dict": g_ema.state_dict(),
                "g_optmizer_state_dict": g_optimizer.state_dict(),
                "d_state_dict": discriminator.state_dict(),
                "d_optmizer_state_dict": d_optimizer.state_dict(),
                "epoch": epoch,
                "stage": 0,
                "current_resolution": current_resolution
            }

            torch.save(checkpoint, args.out + '/checkp.pt')
            
        for epoch in range(config["n_epochs"]):
            print(f"epoch {epoch+1}/{config['n_epochs']}")
            alpha, ema_path_length = train(
                generator,
                discriminator,
                g_ema,
                g_optimizer,
                d_optimizer,
                g_scaler,
                d_scaler,
                precision_scope,
                current_resolution,
                dataloader,
                dataset_size,
                alpha,
                config,
                device,
                fixed_noise,
                args,
                ema_path_length,
                fade_in=False)
            checkpoint = {
                "g_state_dict": generator.state_dict(),
                "g_ema_state_dict": g_ema.state_dict(),
                "g_optmizer_state_dict": g_optimizer.state_dict(),
                "d_state_dict": discriminator.state_dict(),
                "d_optmizer_state_dict": d_optimizer.state_dict(),
                "epoch": epoch,
                "stage": 1,
                "current_resolution": current_resolution
            }

            torch.save(checkpoint, args.out + '/checkp.pt')
        
        current_resolution *= 2


if __name__ == "__main__":
    main()
