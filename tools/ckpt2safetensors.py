import argparse
import os
from safetensors.torch import load_file, save_file
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument("-i", type=str, required=True, help="Path to the input file.")
argparser.add_argument("-o", type=str, required=True, help="Path to the output file.")

args = argparser.parse_args()

weights = torch.load(args.i, map_location="cpu")
weights = weights["state_dict"] if "state_dict" in weights else weights
save_file(weights, args.o)
