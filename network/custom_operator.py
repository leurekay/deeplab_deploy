import torch
import torch.nn as nn
import numpy as np

class FixedDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(FixedDeconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self._init_weights()

    def _init_weights(self):
        # 使用双线性插值权重初始化反卷积层
        bilinear_kernel = self.get_bilinear_kernel(self.deconv.weight.shape[0])
        self.deconv.weight.data.copy_(bilinear_kernel)

        # 固定权重
        for param in self.deconv.parameters():
            param.requires_grad = False

    def get_bilinear_kernel(self, num_channels):
        # 生成双线性插值的卷积核
        factor = (self.deconv.kernel_size[0] + 1) // 2
        if self.deconv.kernel_size[0] % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = (torch.arange(self.deconv.kernel_size[0]).reshape(-1, 1),
              torch.arange(self.deconv.kernel_size[1]).reshape(1, -1))
        filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
        weight = torch.zeros((num_channels, 1, self.deconv.kernel_size[0], self.deconv.kernel_size[1]))
        weight[:, 0, :, :] = filt
        return weight

    def forward(self, x):
        return self.deconv(x)

if __name__=="__main__":
    # 测试代码
    input_tensor = torch.randn(1, 3, 1, 1)
    fixed_deconv = FixedDeconv(3, 16,kernel_size=32,stride=1,padding=0)
    output_tensor = fixed_deconv(input_tensor)
    print("input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)  # torch.Size([1, 3, 48, 48])
