# coding: utf-8
"""
Synthesis waveform for testset

usage: separation.py [options] <checkpoint> <dump-root>

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
from train import build_model, collate_fn, sanity_check
from evaluate import get_data_loader
from nnmnkwii import preprocessing as P
from wavenet_vocoder.util import linear_quantize, inv_linear_quantize


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
               0.625         : 'checkpoints0pt625',
               0.39          : 'checkpoints0pt39',
               0.1           : 'checkpoints0pt0'
}

def main(args):
    model = build_model().to(device)
    model.eval()

    receptive_field = model.receptive_field
    test_data_loader = get_data_loader(args["<dump-root>"], collate_fn)

    (x, y, c, g, input_lengths) = next(iter(test_data_loader))
    # cin_pad = hparams.cin_pad
    # if cin_pad > 0:
    #     c = F.pad(c, pad=(cin_pad, cin_pad), mode="replicate")
    c = c.to(device)
    sanity_check(model, c, g)
    # Write inputs
    x_original_out = inv_linear_quantize(x, hparams.quantize_channels - 1)
    x_original_out = P.inv_mulaw_quantize(x, hparams.quantize_channels - 1)
    sf.write("x_original.wav", x_original_out[0, 0,], hparams.sample_rate)

    # Initialize with noise
    x = torch.FloatTensor(np.random.uniform(-512, 700, size=(1, x.shape[-1] + 1))).to(device)
    # x = F.pad(x, (receptive_field, 0), "constant", 127)
    x.requires_grad = True


    sigmas = [175.9, 110., 68.7,  42.9, 26.8, 16.8, 10.5, 6.55, 4.1, 2.56, 1.6, 1.0, 0.625, 0.39, 0.1]
    start_sigma = 256.
    end_sigma = 0.1

    for idx, sigma in enumerate(sigmas):
        n_steps = 200
        # Bump down a model
        checkpoint_path = join(args["<checkpoint>"], checkpoints[sigma], "checkpoint_latest.pth")
        print("Load checkpoint0 from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

        eta = .02 * (sigma ** 2)
        gamma = 15 * (1.0 / sigma) ** 2

        for i in range(n_steps):
            # Seed with noised up GT, good for unconditional generation
            # x0[0, :receptive_field] = torch.FloatTensor(x0_original[:receptive_field] + np.random.normal(0, sigma, x0_original[:receptive_field].shape)).to(device)
            # x1[0, :receptive_field] = torch.FloatTensor(x1_original[:receptive_field] + np.random.normal(0, sigma, x1_original[:receptive_field].shape)).to(device)

            # Seed with noised up silence
            # x0[0, :receptive_field] = torch.FloatTensor(np.random.normal(127, sigma, x0_original[:receptive_field].shape)).to(device)
            # x1[0, :receptive_field] = torch.FloatTensor(np.random.normal(127, sigma, x1_original[:receptive_field].shape)).to(device)

            # Forward pass
            log_prob, prediction = model.smoothed_loss(x, c=c, sigma=sigma)
            log_prob = torch.sum(log_prob)
            grad = torch.autograd.grad(log_prob, x)[0]
            x_update = eta * grad

            # Langevin step
            epsilon = np.sqrt(2 * eta) * torch.normal(0, 1, size=(1, x.shape[-1]), device=device)
            x_update += epsilon

            with torch.no_grad():
                x += x_update

            if (not i % 20) or (i == (n_steps - 1)): # debugging
                print("--------------")
                print('sigma = {}'.format(sigma))
                print('eta = {}'.format(eta))
                print("i {}".format(i))
                print("Max sample {}".format(
                    abs(x).max()))
                print('Mean sample logpx: {}'.format(log_prob / x.shape[-1]))
                print("Max gradient update: {}".format(eta * abs(grad).max()))

        out = P.inv_mulaw_quantize(x[0, 1:].detach().cpu().numpy(), hparams.quantize_channels - 1)
        # out = inv_linear_quantize(x[0].detach().cpu().numpy(), hparams.quantize_channels - 1)
        out = np.clip(out, -1, 1)
        sf.write("out_{}.wav".format(sigma), out, hparams.sample_rate)


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
