from lrdm.src.low_rank_residual_denoising_diffusion_pytorch import (ResidualDiffusion,Trainer, Unet, UnetRes)
import argparse
from omegaconf import OmegaConf

def train(config_path):
    args = OmegaConf.load(config_path)
    
    #########################
    ######  MODELS
    ########################
    model = UnetRes(
        dim=args.unet_hiddens,
        dim_mults=args.unet_dim_mults,
        share_encoder=args.share_encoder,
        condition=args.condition,
        input_condition=args.input_condition
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.diffusion_timesteps,           # number of steps
        sampling_timesteps=args.sampling_steps,
        objective=args.objective,
        loss_type=args.loss_type,            # L1 or L2
        condition=args.condition,
        sum_scale = args.sum_scale,
        input_condition=args.input_condition,
        input_condition_mask=args.input_condition_mask,
        enable_res_low_rank=args.enable_res_low_rank,
        rank_func=args.rank_func,
    )
    
    
    #########################
    ######  TRAINER
    ########################
    trainer = Trainer(
        diffusion,
        args.data_path,
        train_batch_size=args.train_batch_size,
        num_samples=args.num_samples,
        train_lr=args.train_lr,
        train_num_steps=args.training_num_steps,         # total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=args.ema_decay,                # exponential moving average decay
        amp=False,                        # turn on mixed precision
        convert_image_to="RGB",
        results_folder=args.save_folder,
        condition=args.condition,
        save_and_sample_every=args.save_and_sample_every,
        equalizeHist=False,
        crop_patch=False,
        generation=False
    )
    
    #########################
    ######  TRAINING
    ########################
    trainer.train()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    train(args.config_path)

if __name__ == "__main__":
    main()