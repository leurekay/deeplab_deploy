import torch
import torch.onnx
import network
import onnx
from ade20k import ADE20KSeg
import time




model_name="deeplabv3plus_mobilenet"
weights="checkpoints/best_deeplabv3plus_mobilenet_ade_os8.pth"

# model_name="deeplabv3plus_resnet50"
# weights="checkpoints/best_deeplabv3plus_resnet50_ade_os8.pth"

onnx_output_path=weights.replace(".pth",".onnx")
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
# 假设你有一个准备好的PyTorch模型
model = network.modeling.__dict__[model_name](num_classes=3, output_stride=8)
# network.convert_to_separable_conv(model.classifier)

checkpoint = torch.load(weights, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model = model.eval()
print(model)

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
onnx.checker.check_model(onnx_output_path)
time.sleep(2)

import onnx.utils

# 加载完整的 ONNX 模型
model = onnx.load(onnx_output_path)
print(onnx.helper.printable_graph(model.graph))


# 定义起始层和结束层的名字
input_layer = ['input']  # 起始层名称，通常是模型输入
output_layer = ['/backbone/high_level_features/high_level_features.17/conv/conv.2/Conv_output_0']  # 结束层名称

# 提取子模型
sub_model_path = onnx_output_path.replace(".onnx","_sub.onnx")
onnx.utils.extract_model(onnx_output_path, sub_model_path, input_layer, output_layer)

print(f"Extracted ONNX sub-model has been saved to {sub_model_path}")
