#!/usr/bin/env python
import argparse
import os
import torch

"""
For issue https://github.com/Niger-Volta-LTI/yoruba-adr/issues/14, this script can be used
to prepare a model for release, < 70MB instead of 200MB
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Removes the optim data of PyTorch models")
    parser.add_argument("--model", "-m",
                        help="The model filename (*.pt)", required=True)
    parser.add_argument("--output", "-o",
                        help="The output filename (*.pt)", required=False)
    opt = parser.parse_args()

    if not opt.output:
        opt.output = os.path.splitext(opt.model)[0] + "_release.pt"

    model = torch.load(opt.model)
    model['optim'] = None
    torch.save(model, opt.output)
