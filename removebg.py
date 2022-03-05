
from u2net_test import U2Predict
import argparse

parser = argparse.ArgumentParser(description='Remove the background from the input image.')
parser.add_argument('-i', type=str, help='path to input image')
parser.add_argument('-o', type=str, help='path to output image')

args = parser.parse_args()

U2Predict(img_in=args.i, img_out=args.o)