from datasets import load_dataset
from cleanvision import Imagelab


dataset = load_dataset("cats_vs_dogs", split="train")
dataset