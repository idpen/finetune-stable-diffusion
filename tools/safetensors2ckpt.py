import argparse
import os
from safetensors.torch import load_file
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument("-i", type=str, required=True, help="Path to the input file.")
argparser.add_argument("-o", type=str, required=True, help="Path to the output file.")

args = argparser.parse_args()

weights = load_file(args.i, "cpu")
with open(args.o, "wb") as f:
    torch.save(weights, f)
