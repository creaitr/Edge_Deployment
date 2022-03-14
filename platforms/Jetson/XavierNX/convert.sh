# run a source torchmodel
python detect.py --weights yolov5x.pt --imgsz 640 --source data/images/

# convert torch to onnx file
python torch2onnx.py --weights yolov5x.pt --imgsz 640 --batch -1

# convert onnx to engine files
# dynamic - fp32
trtexec --onnx=yolov5x_i640_b-1.onnx --saveEngine=yolov5x_i640_b-1.engine \
--minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640 --maxShapes=input:1x3x640x640
# dynamic - fp16
trtexec --onnx=yolov5x_i640_b-1.onnx --saveEngine=yolov5x_i640_b-1_fp16.engine \
--minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640 --maxShapes=input:1x3x640x640 --fp16
# dynamic - int8
trtexec --onnx=yolov5x_i640_b-1.onnx --saveEngine=yolov5x_i640_b-1_int8.engine \
--minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640 --maxShapes=input:1x3x640x640 --int8
# dynamic - best
trtexec --onnx=yolov5x_i640_b-1.onnx --saveEngine=yolov5x_i640_b-1_best.engine \
--minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640 --maxShapes=input:1x3x640x640 --best

# run the engine files
# fp32
python detect_trt.py --weights yolov5x.pt --engine yolov5x_i640_b-1.engine --imgsz 640 --source data/images/
# fp16
python detect_trt.py --weights yolov5x.pt --engine yolov5x_i640_b-1_fp16.engine --imgsz 640 --source data/images/
# int8
python detect_trt.py --weights yolov5x.pt --engine yolov5x_i640_b-1_int8.engine --imgsz 640 --source data/images/
# best
python detect_trt.py --weights yolov5x.pt --engine yolov5x_i640_b-1_best.engine --imgsz 640 --source data/images/