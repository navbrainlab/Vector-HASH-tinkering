# Vector-HaSH: 详细说明文档 (中文版)

## 项目概述

这是Nature论文《Episodic and associative memory from spatial scaffolds in the hippocampus》的官方代码实现。该项目实现了Vector-HaSH（Vector Hippocampal Scaffolded Heteroassociative Memory），这是一个新颖的皮层-内嗅皮层-海马网络模型，能够实现高容量通用联想记忆、空间记忆和情景记忆。

## 核心概念

Vector-HaSH的核心思想是利用网格细胞（grid cells）的空间表征作为"脚手架"（scaffold）来组织位置细胞（place cells）的联想记忆。这种方法将空间导航的神经机制扩展到一般的记忆存储和检索任务。

## 文件概览

### 📊 主要实验文件
- `11Rooms_11Maps_Fig_4.ipynb` - 多房间空间映射实验
- `Scaffold_testing_Fig_2.ipynb` - 脚手架网络容量测试  
- `autoencoder_assoc_mem_Fig_3.ipynb` - 自编码器联想记忆对比
- `Full_model_testing_Fig_3.ipynb` - 完整模型测试
- `sequence_autoencoder_Fig_5.ipynb` - 序列记忆实验
- `Sequence_results_VH_and_baselines_Fig_5.ipynb` - 序列结果与基线比较

### 🔧 核心代码模块
- `src/assoc_utils_np_2D.py` - 二维联想记忆核心算法
- `src/seq_utils.py` - 序列学习和处理工具
- `src/capacity_utils.py` - 记忆容量分析工具

### 📈 分析和可视化
- `MTT.py` - 多轨迹测试主程序
- `Plots_for_baseline_item.py` - 基线比较绘图

## 🚀 快速开始

### 运行顺序建议

**第一步：理解基础机制**
```bash
# 1. 首先运行脚手架容量测试，了解基础原理
jupyter notebook Scaffold_testing_Fig_2.ipynb

# 2. 理解联想记忆机制
jupyter notebook autoencoder_assoc_mem_Fig_3.ipynb
```

**第二步：探索空间能力**
```bash
# 3. 多房间空间映射实验
jupyter notebook 11Rooms_11Maps_Fig_4.ipynb

# 4. 完整模型测试
jupyter notebook Full_model_testing_Fig_3.ipynb
```

**第三步：序列和时间动态**
```bash
# 5. 序列记忆实验
jupyter notebook sequence_autoencoder_Fig_5.ipynb

# 6. 序列结果分析
jupyter notebook Sequence_results_VH_and_baselines_Fig_5.ipynb
```

### 环境配置
```python
# 主要依赖包
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import tensorflow as tf  # 仅用于自编码器实验
from tqdm import tqdm
```

### 关键参数设置
```python
# 网格细胞参数
lambdas = [3,4,5,7]  # 网格细胞模块周期
Np = 342             # 位置细胞数量
Ng = 99              # 网格细胞数量 (lambdas平方和)
thresh = 2.0         # 非线性阈值
c = 0.10             # 连接概率
```

## 详细文件结构

### 主要实验脚本（Jupyter Notebooks）

#### 1. `11Rooms_11Maps_Fig_4.ipynb` - 多房间空间映射实验
**功能：** 实现图4中的实验，展示网络在多个不同房间中的空间表征能力。
- **2D GC-PC网络设置：** 建立二维网格细胞-位置细胞网络
- **房间路径生成：** 创建蛇形路径（hairpin pattern）遍历10x10的房间
- **多房间映射：** 生成11个不同位置的房间，展示空间重映射能力
- **序列学习：** 通过MLP分类器学习动作映射，实现路径积分
- **六边形格点可视化：** 在六边形网格上可视化网格场和位置场
- **频率分布分析：** 分析位置细胞在不同房间中的激活模式
- **房间间相关性：** 计算不同房间间位置场的重叠和相关性

#### 2. `Scaffold_testing_Fig_2.ipynb` - 脚手架网络容量测试
**功能：** 对应图2，测试网格细胞脚手架的记忆容量。
- **网格码书生成：** 使用不同的排序策略（最优、螺旋、发夹型）
- **容量分析：** 测试不同参数下的记忆容量
- **理论验证：** 验证理论预测的容量限制

#### 3. `autoencoder_assoc_mem_Fig_3.ipynb` - 自编码器联想记忆对比
**功能：** 对应图3，将Vector-HaSH与传统自编码器进行比较。
- **miniImageNet数据处理：** 加载和预处理miniImageNet数据集
- **自编码器实现：** 使用TensorFlow/Keras实现传统自编码器
- **性能比较：** 比较重建质量和记忆容量
- **噪声鲁棒性测试：** 测试在不同噪声水平下的性能

#### 4. `Full_model_testing_Fig_3.ipynb` - 完整模型测试
**功能：** 测试完整的Vector-HaSH模型性能。
- **完整网络实现：** 集成所有组件的完整测试
- **图像重建：** 测试图像联想记忆能力
- **与基线比较：** 与其他记忆模型的详细比较

#### 5. `sequence_autoencoder_Fig_5.ipynb` & `Sequence_results_VH_and_baselines_Fig_5.ipynb` - 序列记忆实验
**功能：** 对应图5，实现序列记忆和情景记忆功能。
- **序列编码：** 实现时间序列的编码和存储
- **情景记忆：** 模拟情景记忆的形成和检索
- **基线比较：** 与传统序列记忆模型的比较

#### 6. `VectorHASH_minimal_MLP_seq_Fig_5.ipynb` - 最小化MLP序列模型
**功能：** 简化版本的序列学习实现。

#### 7. `Grid_place_tuning_curves_and_additional_expts_Fig1_4_6.ipynb` - 调谐曲线分析
**功能：** 分析网格细胞和位置细胞的调谐特性。

#### 8. `SplitterCells.ipynb` - 分离细胞实验
**功能：** 研究海马分离细胞的特性。

#### 9. `miniimagenet_processing.ipynb` - 数据预处理
**功能：** miniImageNet数据集的预处理和格式转换。

### 核心源代码模块（src/目录）

#### 联想记忆工具
- **`assoc_utils.py`**: 基础联想记忆工具函数
- **`assoc_utils_np.py`**: NumPy优化的联想记忆实现
- **`assoc_utils_np_2D.py`**: 二维环境专用的联想记忆工具
  - `gen_gbook_2d()`: 生成二维网格码书
  - `path_integration_Wgg_2d()`: 二维路径积分矩阵
  - `module_wise_NN_2d()`: 模块化最近邻搜索

#### 序列处理工具
- **`seq_utils.py`**: 序列学习和处理工具
  - `actions()`: 动作编码函数
  - `oneDaction_mapping()`: 一维动作映射
  - 路径编码和解码函数

#### 感知和网格工具
- **`sensory_utils.py`**: 感知输入处理
- **`sensgrid_utils.py`**: 感知-网格交互
- **`sens_pcrec_utils.py`**: 感知-位置细胞循环连接
- **`sens_sparseproj_utils.py`**: 稀疏投影实现
- **`senstranspose_utils.py`**: 感知转置操作

#### 分析工具
- **`capacity_utils.py`**: 记忆容量分析工具
- **`theory_utils.py`**: 理论分析和验证
- **`data_utils.py`**: 数据处理实用函数

### 主要Python脚本

#### 1. `MTT.py` - 多轨迹测试（Multiple Trajectory Testing）
**功能：** 实现多轨迹学习和测试功能。
- **损伤实验：** 模拟海马损伤对记忆的影响
- **图像重建：** 测试在不同损伤程度下的图像重建能力
- **重复学习：** 研究重复暴露对记忆巩固的影响

#### 2. `MTT_plotting_code.py` - MTT结果可视化
**功能：** 生成MTT实验的图表和可视化。

#### 3. `Plots_for_baseline_item.py` & `Plots_for_baseline_seq.py` - 基线比较绘图
**功能：** 生成与基线模型比较的图表。
- 标准Hopfield网络
- 伪逆Hopfield网络
- 稀疏连接Hopfield网络
- 自编码器模型

### 数据文件

- **`pos420by420.mat`, `pos585by585.mat`, `pos60by60.mat`**: MATLAB格式的位置数据，包含不同分辨率的六边形网格位置信息
- **`paths/xy_coords_1000_2.npy`**: 预定义的轨迹坐标数据

## 技术实现细节

### 网络架构
1. **网格细胞层（Grid Cell Layer）**: 使用周期性编码表示空间位置
2. **位置细胞层（Place Cell Layer）**: 通过稀疏连接从网格细胞接收输入
3. **感知输入层（Sensory Input Layer）**: 处理外部感知信息
4. **联想连接（Associative Connections）**: 实现记忆存储和检索

### 关键算法
- **网格细胞编码**: 基于多模块周期性表征
- **路径积分**: 通过矩阵乘法实现位置更新
- **联想记忆**: 使用伪逆学习规则
- **序列学习**: 通过MLP学习动作序列

### 数据预处理
- **图像二值化**: 将彩色图像转换为二值模式
- **噪声注入**: 添加不同类型和强度的噪声
- **序列化**: 将空间和时间信息编码为序列

## 实验重现指南

### 环境配置
```python
# 主要依赖包
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import tensorflow as tf  # 仅用于自编码器实验
```

## 详细文件结构

### 主要实验脚本（Jupyter Notebooks）

## 理论贡献

1. **空间脚手架理论**: 首次提出利用网格细胞的空间表征作为通用记忆组织原理
2. **高容量联想记忆**: 实现了比传统Hopfield网络更高的存储容量
3. **多尺度表征**: 通过多模块网格编码实现不同尺度的空间表征
4. **情景记忆机制**: 将空间导航机制扩展到情景记忆的存储和检索

## 实际应用

该模型为以下领域提供了新的见解：
- **神经科学**: 理解海马体的记忆机制
- **人工智能**: 开发新的记忆网络架构
- **认知科学**: 解释空间和记忆的关系
- **机器学习**: 设计生物启发的联想记忆系统

## 引用

如果使用此代码，请引用原论文：
```
@Article{Chandra&Sharma2025,
author={Chandra, Sarthak and Sharma, Sugandha and Chaudhuri, Rishidev and Fiete, Ila},
title={Episodic and associative memory from spatial scaffolds in the hippocampus},
journal={Nature},
year={2025},
month={Jan},
day={15},
doi={10.1038/s41586-024-08392-y}
}
```

## 联系方式

更多详细信息请参考原论文：https://www.nature.com/articles/s41586-024-08392-y
