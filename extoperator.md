# pytorch扩展算子
扩展算子的主要步骤：
- 1.继承torch.autograd.Function 使用backward和forward函数
- 2.创建nn.Module子模块
