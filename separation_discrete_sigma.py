# coding: utf-8
"""
Synthesis waveform for testset

usage: separation.py [options] <checkpoint0> <checkpoint1>

options:
    --hparams=<parmas>          Hyper parameters [default: ].
    --preset=<json>             Path of preset parameters (json).
    -h, --help                  Show help message.
"""
from docopt import docopt
from os.path import dirname, join, basename, splitext, exists

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import audio
from hparams import hparams
from train import build_model
from nnmnkwii import preprocessing as P
from wavenet_vocoder.util import linear_quantize, inv_linear_quantize

SAMPLE_SIZE = 48000

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Maybe there's a cleaner way of doing this
checkpoints = {
               175.9         : 'checkpoints175pt9',
               110.          : 'checkpoints110pt0',
               68.7          : 'checkpoints68pt7',
               54.3          : 'checkpoints54pt3',
               42.9          : 'checkpoints42pt9',
               34.0          : 'checkpoints34pt0',
               26.8          : 'checkpoints26pt8',
               21.2          : 'checkpoints21pt2',
               16.8          : 'checkpoints16pt8',
               13.3          : 'checkpoints13pt3',
               10.5          : 'checkpoints10pt5',
               8.29          : 'checkpoints8pt29',
               6.55          : 'checkpoints6pt55',
               5.18          : 'checkpoints5pt18',
               4.1           : 'checkpoints4pt1',
               3.24          : 'checkpoints3pt24',
               2.56          : 'checkpoints2pt56',
               1.6           : 'checkpoints1pt6',
               1.0           : 'checkpoints1pt0',
               0.1           : 'checkpoints0pt0'
}

def main(args):
    model0 = build_model().to(device)
    model0.eval()

    model1 = build_model().to(device)
    model1.eval()
    receptive_field = model0.receptive_field

    # Load up some GT samples
    x0_original = np.load("zf882fv0052-wave.npy")
    x0_original = x0_original[300000:300000 + SAMPLE_SIZE]

    x1_original = np.load("p341_048-wave.npy")
    x1_original = x1_original[32000:32000 + SAMPLE_SIZE]

    mixed  = torch.FloatTensor(x0_original + x1_original).reshape(1, -1).to(device)

    # Write inputs
    mixed_out = inv_linear_quantize(mixed[0].detach().cpu().numpy(), hparams.quantize_channels - 1) - 1.0
    mixed_out = np.clip(mixed_out, -1, 1)
    sf.write("mixed.wav", mixed_out, hparams.sample_rate)

    x0_original_out = inv_linear_quantize(x0_original, hparams.quantize_channels - 1)
    sf.write("x0_original.wav", x0_original_out, hparams.sample_rate)

    x1_original_out = inv_linear_quantize(x1_original, hparams.quantize_channels - 1)
    sf.write("x1_original.wav", x1_original_out, hparams.sample_rate)

    # Initialize with noise
    x0 = torch.FloatTensor(np.random.uniform(-512, 700, size=(1, SAMPLE_SIZE))).to(device)
    x0 = F.pad(x0, (receptive_field, 0), "constant", 127)
    x0.requires_grad = True

    x1 = torch.FloatTensor(np.random.uniform(-512, 700, size=(1, SAMPLE_SIZE))).to(device)
    x1 = F.pad(x1, (receptive_field, 0), "constant", 127)
    x1.requires_grad = True

    # Initialize with noised GT
    # x0[0, receptive_field:] = torch.FloatTensor(x0_original + np.random.normal(0, 256., x0_original.shape)).to(device)
    # x1[0, receptive_field:] = torch.FloatTensor(x1_original + np.random.normal(0, 256., x1_original.shape)).to(device)

    sigmas = [175.9, 110., 68.7, 54.3, 42.9, 34.0, 26.8, 21.2, 16.8, 10.5, 6.55, 4.1, 2.56, 1.6, 1.0, 0.1]
    # n_steps_each = [500, 1000, 1000, 1000, 2000, 2000, 2000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]
    start_sigma = 256.
    end_sigma = 0.1

    for idx, sigma in enumerate(sigmas):
        n_steps = 60
        # Bump down a model
        checkpoint_path0 = join(args["<checkpoint0>"], checkpoints[sigma], "checkpoint_latest.pth")
        checkpoint_path1 = join(args["<checkpoint1>"], checkpoints[sigma], "checkpoint_latest.pth")
        print("Load checkpoint0 from {}".format(checkpoint_path0))
        checkpoint0 = torch.load(checkpoint_path0)
        checkpoint1 = torch.load(checkpoint_path1)
        model0.load_state_dict(checkpoint0["state_dict"])
        model1.load_state_dict(checkpoint1["state_dict"])

        eta = .05 * (sigma ** 2)
        gamma = 15 * (1.0 / sigma) ** 2

        for i in range(n_steps):
            # Seed with noised up GT, good for unconditional generation
            # x0[0, :receptive_field] = torch.FloatTensor(x0_original[:receptive_field] + np.random.normal(0, sigma, x0_original[:receptive_field].shape)).to(device)
            # x1[0, :receptive_field] = torch.FloatTensor(x1_original[:receptive_field] + np.random.normal(0, sigma, x1_original[:receptive_field].shape)).to(device)

            # Seed with noised up silence
            x0[0, :receptive_field] = torch.FloatTensor(np.random.normal(127, sigma, x0_original[:receptive_field].shape)).to(device)
            x1[0, :receptive_field] = torch.FloatTensor(np.random.normal(127, sigma, x1_original[:receptive_field].shape)).to(device)

            # Forward pass
            log_prob, prediction0 = model0.smoothed_loss(x0, sigma=sigma)
            log_prob0 = torch.sum(log_prob[:, (receptive_field - 1):])
            grad0 = torch.autograd.grad(log_prob0, x0)[0]
            x0_update = eta * grad0[:, receptive_field:]

            log_prob, prediction1 = model1.smoothed_loss(x1, sigma=sigma)
            log_prob1 = torch.sum(log_prob[:, (receptive_field - 1):])
            grad1 = torch.autograd.grad(log_prob1, x1)[0]
            x1_update = eta * grad1[:, receptive_field:]

            # Langevin step
            epsilon0 = np.sqrt(2 * eta) * torch.normal(0, 1, size=(1, SAMPLE_SIZE), device=device)
            x0_update += epsilon0

            epsilon1 = np.sqrt(2 * eta) * torch.normal(0, 1, size=(1, SAMPLE_SIZE), device=device)
            x1_update += epsilon1

            # Reconstruction step
            x0_update -= eta * gamma * (x0[:, receptive_field:] + x1[:, receptive_field:] - mixed)
            x1_update -= eta * gamma * (x0[:, receptive_field:] + x1[:, receptive_field:] - mixed)

            with torch.no_grad():
                x0[:, receptive_field:] += x0_update
                x1[:, receptive_field:] += x1_update

            if (not i % 20) or (i == (n_steps - 1)): # debugging
                print("--------------")
                print('sigma = {}'.format(sigma))
                print('eta = {}'.format(eta))
                print("i {}".format(i))
                print("Max sample {}".format(
                    abs(x0).max()))
                print('Mean sample logpx: {}'.format(log_prob0 / SAMPLE_SIZE))
                print('Mean sample logpy: {}'.format(log_prob1 / SAMPLE_SIZE))
                print("Max gradient update: {}".format(eta * abs(grad0).max()))
                print("Reconstruction: {}".format(abs(x0[:, receptive_field:] + x1[:, receptive_field:] - mixed).mean()))


        # out0 = P.inv_mulaw_quantize(x0[0].detach().cpu().numpy(), hparams.quantize_channels - 1)
        out0 = inv_linear_quantize(x0[0].detach().cpu().numpy(), hparams.quantize_channels - 1)
        out0 = np.clip(out0, -1, 1)
        sf.write("out0_{}.wav".format(sigma), out0, hparams.sample_rate)

        out1 = inv_linear_quantize(x1[0].detach().cpu().numpy(), hparams.quantize_channels - 1)
        out1 = np.clip(out1, -1, 1)
        sf.write("out1_{}.wav".format(sigma), out1, hparams.sample_rate)

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    args = docopt(__doc__)

    # Load preset if specified
    if args["--preset"] is not None:
        with open(args["--preset"]) as f:
            hparams.parse_json(f.read())
    else:
        hparams_json = join(dirname(args["<checkpoint1>"]), "hparams.json")
        if exists(hparams_json):
            print("Loading hparams from {}".format(hparams_json))
            with open(hparams_json) as f:
                hparams.parse_json(f.read())

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    main(args)
