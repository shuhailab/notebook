# pytorch扩展算子
扩展算子的主要步骤：
- 1.继承torch.autograd.Function 使用backward和forward函数
- 2.创建nn.Module子模块
- 扩展算子并且导出onnx文件 code/extop_Linear.py
