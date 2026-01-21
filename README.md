# MCB-U
CoordAtt-U
CoordAtt-U 基于Coordinate Attention，在原本的方向注意力（H/W）基础上，引入 MCB 损失产生的类别权重 w_mcb 作为训练信号：当 w_mcb 波动更大（std 更大）时，自动增强注意力强度，让模型更聚焦关键特征。同时加入残差下限，保证不会把原始特征压没，训练更稳定。
MCB-Convex
MCB-Convex：它根据当前 batch 中各类别预测概率的波动（std）计算类别权重，并用 softmax + 最小权重约束（w_min）+ EMA 平滑 来稳定训练。最终让“更难/更不稳定的类别”获得更大损失权重，从而提升整体均衡学习效果
