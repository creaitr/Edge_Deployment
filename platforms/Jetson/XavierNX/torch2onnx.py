import sys
import os
import argparse
from pathlib import Path
import json
import math
import numpy as np

import torch

# Below modules are from ultralytics/yolov5 github:
# https://www.google.com/search?q=yolov5+github&oq=yolo&aqs=chrome.0.69i59l2j69i57j69i59j69i60l4.1004j0j7&sourceid=chrome&ie=UTF-8
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device


def convert_torch2onnx(weights, batch, imgsz):
    print("Converting to onnx and running demo ...")

    # Initialize
    #device = select_device('') # cuda device, i.e. 0 or 0,1,2,3 or cpu
    device = torch.device('cpu')

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.eval()
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    input_names = ['input']
    output_names = ['out0', 'out1', 'out2', 'out3']
    onnx_file_name = '{}_i{}_b{}.onnx'.format(weights.split('.pt')[0], imgsz, batch)

    if batch > 0: # static model
        x = torch.randn((batch, 3, imgsz, imgsz), requires_grad=False, device=device)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None,
                          verbose=False)

    else: # dynamic model
        x = torch.randn((1, 3, imgsz, imgsz), requires_grad=False, device=device)

        dynamic_axes = {
            "input": {0: "batch_size"},
            "out0": {0: "batch_size"},
            "out1": {0: "batch_size"},
            "out2": {0: "batch_size"},
            "out3": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          verbose=False)
    
    print('Onnx model exporting done')
    return onnx_file_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images/bus.jpg', help='file/dir/URL/image.png')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--batch', type=int, default=-1, help='size of batch')
    args = parser.parse_args()

    onnx_path = convert_torch2onnx(args.weights, args.batch, args.imgsz)


### Future Work ###
# Run the converted .onnx model with ONNX package

# import onnx
# import onnxruntime
#
# def do_inference(onnx_path, image):
#     # session = onnx.load(onnx_path)
#     session = onnxruntime.InferenceSession(onnx_path)
#
#     # Check image size
#     assert session.get_inputs()[0].shape == image.shape
#
#     # Compute
#     input_name = session.get_inputs()[0].name
#     outputs = session.run(None, {input_name: img_in})
