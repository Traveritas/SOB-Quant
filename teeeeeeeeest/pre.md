(base) PS D:\Documents\HTA\College\Projects\quant> (D:\Anaconda\shell\condabin\conda-hook.ps1) ; (conda activate deeplearning)
(deeplearning) PS D:\Documents\HTA\College\Projects\quant> python .\quant_effect_test_v2.py               
======================================================================
=== 1) 训练基线 Float32 模型 ===
======================================================================
Epoch 1 | step    0 | seen     0 | loss 2.2945
Epoch 1 | step  100 | seen 12800 | loss 0.3588
Epoch 1 | step  200 | seen 25600 | loss 0.3257

======================================================================
=== 2) 评估基线模型（含重复测速）===
======================================================================
基线准确率: 90.34%
基线平均 batch 时延: 0.6036 ± 0.0395 ms
基线平均 sample 时延: 0.001207 ± 0.000079 ms
基线测速配置: warmup=5, batches=20, repeats=20
基线参数体积: 0.3882 MB (3256640 bits)

======================================================================
=== 3) 应用你的量化方法（fc1 + fc2） ===
======================================================================
层 fc1: MSE=0.000020, MAE=0.003398
  码本使用统计: {-3.0: 2385, -1.0: 47860, 1.0: 47691, 3.0: 2416}
  维度: d=784, m=128
层 fc2: MSE=0.000045, MAE=0.005329
  码本使用统计: {-1.0: 632, 1.0: 648}
  维度: d=128, m=10

======================================================================
=== 4) 评估量化后模型（含重复测速）===
======================================================================
量化后准确率: 90.44%
量化后平均 batch 时延: 0.7037 ± 0.0474 ms
量化后平均 sample 时延: 0.001407 ± 0.000095 ms
量化后测速配置: warmup=5, batches=20, repeats=20
量化后参数体积(float32): 0.3882 MB
量化表示估算体积(位级): 2.4355 MB (20430272 bits)

======================================================================
=== 5) 量化效果详细总结 ===
======================================================================
【精度指标】
  基线vs量化: 90.34% vs 90.44%
  精度变化: -0.10% (基线-量化)

【速度指标】
  基线 batch 时延: 0.6036 ms
  量化 batch 时延: 0.7037 ms
  时延变化: +0.1001 ms (+16.58%)

【存储指标】
  基线参数体积: 0.3882 MB (Float32 native)
  量化后直接写回: 0.3882 MB (仍为 Float32)
  量化表示估算: 2.4355 MB (离散索引+分解参数)
  存储体积变化(估算): +2.0473 MB
  理论压缩率(基线/量化表示): 0.16x

======================================================================
说明
======================================================================
• 基线参数体积不变: 当前实现是把量化后的重建权重写回 float32 张量
• 位级估算体积: 反映按离散索引+分解参数(U,Lambda,Z)存储的理论体积
• 理论压缩率: 只针对量化层(fc1, fc2)，未计入输入层和输出层
• 测速结果: 已包含 warmup 排除缓存效应，多次运行求均值降低波动
(deeplearning) PS D:\Documents\HTA\College\Projects\quant> python .\quant_effect_resnet18.py
使用设备: cuda
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  显存: 8.59 GB

======================================================================
=== 1) 创建并训练 ResNet-18 基线模型 ===
======================================================================
D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.  
  warnings.warn(
D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
模型参数总数: 11,175,370 (11.18M)
Epoch 1 | step    0 | seen     0 | loss 2.4674
Epoch 1 | step   50 | seen 12800 | loss 0.2721
Epoch 1 | step  100 | seen 25600 | loss 0.1917
Epoch 1 | step  150 | seen 38400 | loss 0.1552
Epoch 1 | step  200 | seen 51200 | loss 0.1351
Epoch 2 | step    0 | seen     0 | loss 0.0293
Epoch 2 | step   50 | seen 12800 | loss 0.0504
Epoch 2 | step  100 | seen 25600 | loss 0.0478
Epoch 2 | step  150 | seen 38400 | loss 0.0491
Epoch 2 | step  200 | seen 51200 | loss 0.0501

======================================================================
=== 2) 评估基线模型（含重复测速）===
======================================================================
基线准确率: 97.45%
基线平均 batch 时延: 8.8373 ± 2.5864 ms
基线平均 sample 时延: 0.017675 ± 0.005173 ms
基线测速配置: warmup=5, batches=20, repeats=20
基线参数体积: 42.6307 MB (357.61M bits)

======================================================================
=== 3) 应用量化到所有 FC 层 ===
======================================================================
  跳过层 fc（太小: 512x10）

量化了 0 层:

======================================================================
=== 4) 评估量化后模型（含重复测速）===
======================================================================
量化后准确率: 97.45%
量化后平均 batch 时延: 8.6759 ± 2.5051 ms
量化后平均 sample 时延: 0.017352 ± 0.005010 ms
量化后参数体积(float32): 42.6307 MB
量化表示估算体积(位级): 42.6307 MB (357.61M bits)

======================================================================
=== 5) 量化效果详细总结 ===
======================================================================
【精度指标】
  基线: 97.45% | 量化: 97.45%
  精度变化: +0.00% (基线-量化)

【速度指标】
  基线 batch 时延: 8.8373 ms
  量化 batch 时延: 8.6759 ms
  时延变化: -0.1614 ms (-1.83%)

【存储指标】
  基线参数体积: 42.6307 MB
  量化表示估算: 42.6307 MB
  存储体积变化: +0.0000 MB
  理论压缩率(基线/量化表示): 1.00x

======================================================================
说明
======================================================================
• 模型: ResNet-18，11.18M 参数（比 SimpleNet 大 ~30 倍）
• 量化策略: 所有 FC 层（跳过参数 < 32x32 的层）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit)
• 测速: warmup=5, 20个 batch，重复 20 次，输出均值±标准差
• GPU: 已启用 CUDA 加速（如果可用）
(deeplearning) PS D:\Documents\HTA\College\Projects\quant> python .\quant_effect_resnet18_v2.py
使用设备: cuda
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  显存: 8.59 GB

======================================================================
=== 1) 创建并训练 ResNet-18 基线模型（含大 FC 层）===
======================================================================
D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.  
  warnings.warn(
D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
模型参数总数: 11,705,802 (11.71M)
  FC 层: fc.0 -> in=512, out=1024
  FC 层: fc.2 -> in=1024, out=10
Epoch 1 | step    0 | seen     0 | loss 2.3079
Epoch 1 | step   50 | seen 12800 | loss 0.3283
Epoch 1 | step  100 | seen 25600 | loss 0.2159
Epoch 1 | step  150 | seen 38400 | loss 0.1743
Epoch 1 | step  200 | seen 51200 | loss 0.1511
Epoch 2 | step    0 | seen     0 | loss 0.0290
Epoch 2 | step   50 | seen 12800 | loss 0.0512
Epoch 2 | step  100 | seen 25600 | loss 0.0536
Epoch 2 | step  150 | seen 38400 | loss 0.0546
Epoch 2 | step  200 | seen 51200 | loss 0.0537

======================================================================
=== 2) 评估基线模型（含重复测速）===
======================================================================
基线准确率: 98.70%
基线平均 batch 时延: 8.6800 ± 2.1691 ms
基线平均 sample 时延: 0.017360 ± 0.004338 ms
基线测速配置: warmup=5, batches=20, repeats=20
基线参数体积: 44.6541 MB (374.59M bits)

======================================================================
=== 3) 应用量化到所有 FC 层 ===
======================================================================
  量化层 fc.0 (512x1024)... ✓
  跳过层 fc.2（太小: 1024x10）

量化了 1 层:
  fc.0: MSE=0.000069, MAE=0.006617, d=512, m=1024

======================================================================
=== 4) 评估量化后模型（含重复测速）===
======================================================================
Traceback (most recent call last):
  File "D:\Documents\HTA\College\Projects\quant\quant_effect_resnet18_v2.py", line 474, in <module>
    main()
    ~~~~^^
  File "D:\Documents\HTA\College\Projects\quant\quant_effect_resnet18_v2.py", line 407, in main
    quant_acc = evaluate_accuracy(quant_model, test_loader, device)
  File "D:\Documents\HTA\College\Projects\quant\quant_effect_resnet18_v2.py", line 113, in evaluate_accuracy
    output = model(data)
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\resnet.py", line 285, in forward
    return self._forward_impl(x)
           ~~~~~~~~~~~~~~~~~~^^^
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\resnet.py", line 280, in _forward_impl
    x = self.fc(x)
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torch\nn\modules\container.py", line 240, in forward
    input = module(input)
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torch\nn\modules\module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torch\nn\modules\module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Anaconda\envs\deeplearning\Lib\site-packages\torch\nn\modules\linear.py", line 125, in forward 
    return F.linear(input, self.weight, self.bias)
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat2 in method wrapper_CUDA_addmm)
(deeplearning) PS D:\Documents\HTA\College\Projects\quant> python .\quant_effect_resnet18_v2.py
使用设备: cuda
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  显存: 8.59 GB

======================================================================
=== 1) 创建并训练 ResNet-18 基线模型（含大 FC 层）===
======================================================================
D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.  
  warnings.warn(
D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
模型参数总数: 11,705,802 (11.71M)
  FC 层: fc.0 -> in=512, out=1024
  FC 层: fc.2 -> in=1024, out=10
Epoch 1 | step    0 | seen     0 | loss 2.3079
Epoch 1 | step   50 | seen 12800 | loss 0.3246
Epoch 1 | step  100 | seen 25600 | loss 0.2159
Epoch 1 | step  150 | seen 38400 | loss 0.1755
Epoch 1 | step  200 | seen 51200 | loss 0.1513
Epoch 2 | step    0 | seen     0 | loss 0.0623
Epoch 2 | step   50 | seen 12800 | loss 0.0531
Epoch 2 | step  100 | seen 25600 | loss 0.0589
Epoch 2 | step  150 | seen 38400 | loss 0.0576
Epoch 2 | step  200 | seen 51200 | loss 0.0564

======================================================================
=== 2) 评估基线模型（含重复测速）===
======================================================================
基线准确率: 98.12%
基线平均 batch 时延: 8.6983 ± 2.1819 ms
基线平均 sample 时延: 0.017397 ± 0.004364 ms
基线测速配置: warmup=5, batches=20, repeats=20
基线参数体积: 44.6541 MB (374.59M bits)

======================================================================
=== 3) 应用量化到所有 FC 层 ===
======================================================================
  量化层 fc.0 (512x1024)... ✓
  跳过层 fc.2（太小: 1024x10）

量化了 1 层:
  fc.0: MSE=0.000069, MAE=0.006616, d=512, m=1024

======================================================================
=== 4) 评估量化后模型（含重复测速）===
======================================================================
量化后准确率: 98.14%
量化后平均 batch 时延: 8.8046 ± 2.5774 ms
量化后平均 sample 时延: 0.017609 ± 0.005155 ms
量化后参数体积(float32): 44.6541 MB
量化表示估算体积(位级): 43.7811 MB (367.26M bits)

======================================================================
=== 5) 量化效果详细总结 ===
======================================================================
【精度指标】
  基线: 98.12% | 量化: 98.14%
  精度变化: -0.02% (基线-量化)

【速度指标】
  基线 batch 时延: 8.6983 ms
  量化 batch 时延: 8.8046 ms
  时延变化: +0.1063 ms (+1.22%)

【存储指标】
  基线参数体积: 44.6541 MB
  量化表示估算: 43.7811 MB
  存储体积变化: -0.8730 MB
  理论压缩率(基线/量化表示): 1.02x

======================================================================
说明
======================================================================
• 模型: ResNet-18 + 大 FC 隐层 (512x1024, 1024x10)，11.71M 参数
• 量化策略: 所有 FC 层（跳过参数 < 32x32 的层）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit)
• 测速: warmup=5, 20个 batch，重复 20 次，输出均值±标准差
• GPU: RTX4060，已启用 CUDA 加速
(deeplearning) PS D:\Documents\HTA\College\Projects\quant> python .\quant_effect_resnet18_conv.py
使用设备: cuda
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  显存: 8.59 GB

======================================================================
=== 1) 创建并训练 ResNet-18 基线模型 ===
======================================================================
D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.  
  warnings.warn(
D:\Anaconda\envs\deeplearning\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
模型参数总数: 11,705,802 (11.71M)
Epoch 1 | step    0 | seen     0 | loss 2.3079
Epoch 1 | step   50 | seen 12800 | loss 0.3279
Epoch 1 | step  100 | seen 25600 | loss 0.2179
Epoch 1 | step  150 | seen 38400 | loss 0.1760
Epoch 1 | step  200 | seen 51200 | loss 0.1531
Epoch 2 | step    0 | seen     0 | loss 0.0293
Epoch 2 | step   50 | seen 12800 | loss 0.0502
Epoch 2 | step  100 | seen 25600 | loss 0.0565
Epoch 2 | step  150 | seen 38400 | loss 0.0584
Epoch 2 | step  200 | seen 51200 | loss 0.0571

======================================================================
=== 2) 评估基线模型（含重复测速）===
======================================================================
基线准确率: 98.16%
基线平均 batch 时延: 8.9658 ± 2.7702 ms
基线平均 sample 时延: 0.017932 ± 0.005540 ms
基线参数体积: 44.6541 MB (374.59M bits)

======================================================================
=== 3) 应用量化到所有卷积层和 FC 层 ===
======================================================================
  量化层 conv1 (64, 1, 7, 7)... ✓
  量化层 layer1.0.conv1 (64, 64, 3, 3)... ✓
  量化层 layer1.0.conv2 (64, 64, 3, 3)... ✓
  量化层 layer1.1.conv1 (64, 64, 3, 3)... ✓
  量化层 layer1.1.conv2 (64, 64, 3, 3)... ✓
  量化层 layer2.0.conv1 (128, 64, 3, 3)... ✓
  量化层 layer2.0.conv2 (128, 128, 3, 3)... ✓
  量化层 layer2.0.downsample.0 (128, 64, 1, 1)... ✓
  量化层 layer2.1.conv1 (128, 128, 3, 3)... ✓
  量化层 layer2.1.conv2 (128, 128, 3, 3)... ✓
  量化层 layer3.0.conv1 (256, 128, 3, 3)... ✓
  量化层 layer3.0.conv2 (256, 256, 3, 3)... ✓
  量化层 layer3.0.downsample.0 (256, 128, 1, 1)... ✓
  量化层 layer3.1.conv1 (256, 256, 3, 3)... ✓
  量化层 layer3.1.conv2 (256, 256, 3, 3)... ✓
  量化层 layer4.0.conv1 (512, 256, 3, 3)... ✓
  量化层 layer4.0.conv2 (512, 512, 3, 3)... ✓
  量化层 layer4.0.downsample.0 (512, 256, 1, 1)... ✓
  量化层 layer4.1.conv1 (512, 512, 3, 3)... ✓
  量化层 layer4.1.conv2 (512, 512, 3, 3)... ✓
  量化层 fc.0 (1024, 512)... ✓
  跳过层 fc.2（太小: 1024x10）

量化了 21 层
  conv1: d=64, m=49, params=3,136, MSE=0.000471
  layer1.0.conv1: d=64, m=576, params=36,864, MSE=0.000464
  layer1.0.conv2: d=64, m=576, params=36,864, MSE=0.000449
  layer1.1.conv1: d=64, m=576, params=36,864, MSE=0.000445
  layer1.1.conv2: d=64, m=576, params=36,864, MSE=0.000445
  layer2.0.conv1: d=128, m=576, params=73,728, MSE=0.000217
  layer2.0.conv2: d=128, m=1152, params=147,456, MSE=0.000234
  layer2.0.downsample.0: d=128, m=64, params=8,192, MSE=0.000833
  layer2.1.conv1: d=128, m=1152, params=147,456, MSE=0.000234
  layer2.1.conv2: d=128, m=1152, params=147,456, MSE=0.000231
  layer3.0.conv1: d=256, m=1152, params=294,912, MSE=0.000113
  layer3.0.conv2: d=256, m=2304, params=589,824, MSE=0.000118
  layer3.0.downsample.0: d=256, m=128, params=32,768, MSE=0.000439
  layer3.1.conv1: d=256, m=2304, params=589,824, MSE=0.000117
  layer3.1.conv2: d=256, m=2304, params=589,824, MSE=0.000116
  layer4.0.conv1: d=512, m=2304, params=1,179,648, MSE=0.000054
  layer4.0.conv2: d=512, m=4608, params=2,359,296, MSE=0.000055
  layer4.0.downsample.0: d=512, m=256, params=131,072, MSE=0.000226
  layer4.1.conv1: d=512, m=4608, params=2,359,296, MSE=0.000055
  layer4.1.conv2: d=512, m=4608, params=2,359,296, MSE=0.000055
  fc.0: d=512, m=1024, params=524,288, MSE=0.000069
量化参数占比: 99.82%

======================================================================
=== 4) 评估量化后模型（含重复测速）===
======================================================================
量化后准确率: 83.40%
量化后平均 batch 时延: 8.3672 ± 2.2896 ms
量化后平均 sample 时延: 0.016734 ± 0.004579 ms
量化后参数体积(float32): 44.6541 MB
量化表示估算体积(位级): 10.5264 MB (88.30M bits)

======================================================================
=== 5) 量化效果详细总结 ===
======================================================================
【精度指标】
  基线: 98.16% | 量化: 83.40%
  精度变化: +14.76% (基线-量化)

【速度指标】
  基线 batch 时延: 8.9658 ms
  量化 batch 时延: 8.3672 ms
  精度变化: +14.76% (基线-量化)

【速度指标】
  基线 batch 时延: 8.9658 ms
  量化 batch 时延: 8.3672 ms
【速度指标】
  基线 batch 时延: 8.9658 ms
  量化 batch 时延: 8.3672 ms
  时延变化: -0.5986 ms (-6.68%)

【存储指标】
  基线参数体积: 44.6541 MB
  时延变化: -0.5986 ms (-6.68%)

【存储指标】
  基线参数体积: 44.6541 MB
【存储指标】
  基线参数体积: 44.6541 MB
  量化表示估算: 10.5264 MB
  量化表示估算: 10.5264 MB
  存储体积变化: -34.1277 MB
  存储体积变化: -34.1277 MB
  理论压缩率(基线/量化表示): 4.24x

======================================================================
======================================================================
说明
======================================================================
• 模型: ResNet-18，11.71M 参数
======================================================================
• 模型: ResNet-18，11.71M 参数
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit) - 共享码本
• 卷积层处理: 4D -> 2D reshape -> 量化 -> reshape 回 4D
• 测速: warmup=5, 20个 batch，重复 20 次，输出均值±标准差
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit) - 共享码本
• 卷积层处理: 4D -> 2D reshape -> 量化 -> reshape 回 4D
======================================================================
======================================================================
======================================================================
• 模型: ResNet-18，11.71M 参数
======================================================================
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit) - 共享码本
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit) - 共享码本
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit) - 共享码本
======================================================================
• 模型: ResNet-18，11.71M 参数
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit) - 共享码本
======================================================================
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit) - 共享码本
• 模型: ResNet-18，11.71M 参数
• 量化策略: 所有卷积层 + FC 层（共 21 层，占 99.82% 参数）
• 分解表示: U(d²) + Lambda(d) + Z 索引(2-bit) - 共享码本
• 卷积层处理: 4D -> 2D reshape -> 量化 -> reshape 回 4D
• 测速: warmup=5, 20个 batch，重复 20 次，输出均值±标准差
• GPU: RTX4060，已启用 CUDA 加速