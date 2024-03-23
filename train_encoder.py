from read_image_data import read_image_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--training_images")
parser.add_argument("-e", "--encoder-model")
parser.add_argument("-o", "--output-file")

