import torch
import torch.onnx
import network

from ade20k import ADE20KSeg




# model_name="deeplabv3plus_mobilenet"
# weights="checkpoints/best_deeplabv3plus_mobilenet_ade_os8.pth"

model_name="deeplabv3plus_resnet50"
weights="checkpoints/best_deeplabv3plus_resnet50_ade_os8.pth"

onnx_output_path=weights.replace(".pth",".onnx")
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
# 假设你有一个准备好的PyTorch模型
model = network.modeling.__dict__[model_name](num_classes=3, output_stride=8)
# network.convert_to_separable_conv(model.classifier)

checkpoint = torch.load(weights, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)

# 准备一个输入样本
x = torch.randn(1, 3, 480, 640)
 
# 导出模型到ONNX格式
torch.onnx.export(model,               # 模型的实例
                  x,                   # 模型的输入
                  onnx_output_path,        # 导出的ONNX模型的文件名
                  export_params=True,  # 是否导出模型参数
                  opset_version=12,    # ONNX操作集版本
                  do_constant_folding=True,  # 是否进行常量折叠
                  input_names = ['input'],   # 模型输入的名称
                  output_names = ['output'], # 模型输出的名称
                #   dynamic_axes={'input' : {0: "batch", 2: "height", 3: "width"},    # 动态轴
                #                 'output' : {0: "batch", 2: "mask_height", 3: "mask_width"}}
                    # dynamic_axes={'input' : {0: "batch"},    # 动态轴
                    #                 'output' : {0: "batch"}}
                                )


# def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr("ONNX:")):
#     """Exports a YOLOv5 model to ONNX format with dynamic axes and optional simplification."""
#     import onnx

#     f = str(file.with_suffix(".onnx"))

#     output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"]
#     if dynamic:
#         dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
#         if isinstance(model, SegmentationModel):
#             dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
#             dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
#         elif isinstance(model, DetectionModel):
#             dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)

#     torch.onnx.export(
#         model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
#         im.cpu() if dynamic else im,
#         f,
#         verbose=False,
#         opset_version=opset,
#         do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
#         input_names=["images"],
#         output_names=output_names,
#         dynamic_axes=dynamic or None,
#     )

#     # Checks
#     model_onnx = onnx.load(f)  # load onnx model
#     onnx.checker.check_model(model_onnx)  # check onnx model

#     # Metadata
#     d = {"stride": int(max(model.stride)), "names": model.names}
#     for k, v in d.items():
#         meta = model_onnx.metadata_props.add()
#         meta.key, meta.value = k, str(v)
#     onnx.save(model_onnx, f)

#     # Simplify
#     if simplify:
#         try:
#             cuda = torch.cuda.is_available()
#             check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.4.1"))
#             import onnxsim

#             LOGGER.info(f"{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...")
#             model_onnx, check = onnxsim.simplify(model_onnx)
#             assert check, "assert check failed"
#             onnx.save(model_onnx, f)
#         except Exception as e:
#             LOGGER.info(f"{prefix} simplifier failure: {e}")
#     return f, model_onnx