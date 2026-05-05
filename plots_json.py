from datetime import datetime
import json
import os

from plots.plots import save_loss_plot, save_loss_plot_lambda

_lambda = 700

actual_dir = os.path.dirname(os.path.realpath(__file__))

checkpoint_dir = os.path.join(actual_dir, 'experiments')

# CREO CARTELLA STAMPA GRAFICI
log_dir = os.path.join(checkpoint_dir, 'stampa_grafici_{}'.format(datetime.now().strftime("%d-%m-%Y--%H-%M")))

if not os.path.exists(log_dir):       
    os.mkdir(log_dir)


exp = "experiment_27-04-2026--14-18"


experimentPath = os.path.join(actual_dir,"experiments", exp)

with open(os.path.join(experimentPath, "losses_batch.json"), "r") as f:
    losses_history = json.load(f)

# Salvo storico delle loss medie di ogni epoca del training
with open(os.path.join(experimentPath, "losses_epochs.json"), "r") as f:
    losses_history_epochs = json.load(f)

losses = {
          "gen_total",
          "gen_gan",
          "gen_l1",
          "disc_total",
          "disc_real",
          "disc_fake"
}

# GENERATOR tutte
save_loss_plot(
    losses_history_epochs,
    ["gen_total", "gen_gan", "gen_l1"],
    "Generator Losses",
    "generator_losses.png",
    log_dir
)

# GENERATOR no total
save_loss_plot(
    losses_history_epochs,
    ["gen_gan", "gen_l1"],
    "Generator Losses (no total)",
    "generator_losses_no_total.png",
    log_dir
)

# DISCRIMINATOR
save_loss_plot(
    losses_history_epochs,
    ["disc_total", "disc_real", "disc_fake"],
    "Discriminator Losses",
    "discriminator_losses.png",
    log_dir
)

# DISCRIMINATOR e GENERATOR
save_loss_plot(
    losses_history_epochs,
    ["gen_total", "disc_total"],
    "Discriminator and Generator Losses",
    "disc_gen_tot_losses.png",
    log_dir
)

# DISCRIMINATOR e GENERATOR loss della gan 
save_loss_plot(
    losses_history_epochs,
    ["gen_gan", "disc_fake"],
    "Discriminator and Generator GAN Losses",
    "disc_gen_gan_losses.png",
    log_dir
)

# DISCRIMINATOR e GENERATOR loss della gan 
save_loss_plot_lambda(
    losses_history_epochs,
    ["gen_l1", "gen_gan"],
    "Generator L1*lambda e GAN Losses",
    "gen_l1_lambda_gan_losses.png",
    log_dir,
    _lambda = _lambda
)

