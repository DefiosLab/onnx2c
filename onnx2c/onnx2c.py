#!/usr/bin/env python3
import argparse
import onnx2c
import onnx 

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--onnx', type=str, required=True,help="input onnx path")
    parser.add_argument('--input_data', type=str, help="Include the input data in the header.Please specify an npy file.")

    args = parser.parse_args()
    return args
def main():
    args = arg_parser()
    # Generate C
    model = onnx.load(args.onnx)
    gen = onnx2c.Generator(model)
    
if __name__ == "__main__":
    main()
