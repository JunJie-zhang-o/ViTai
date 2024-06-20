## 基于GelSight和HandE夹爪的自适应物体夹取

### 简介

基于GelSight视触觉传感器和HandE夹爪，通过图像信息的形变信息实现对目标物体的自适应夹取，夹取对象甚至可以是一些比较柔软的物体或不规则物体，比如鸡蛋、树叶、薯片、水果等。

### 效果展示



### 文件结构 

```
.
├── gelsight		--图像读取和深度图重建
├── main1.py		--使用触觉图像和深度图像自适应夹取
├── main2.py		--仅使用触觉图像夹取
├── nnmini.pt
├── README.MD
├── requirements.txt
├── Robotiq			--手爪控制代码
└── utils

```

### 如何使用

#### 环境准备

1. 创建并激活虚拟环境

```shell
conda env create --name ViTai
conda activate ViTai
```
2. 按照requiremwnts.txt配置环境
```shell
pip install -r requirements.txt
```

#### 相关配置



#### 启动

```python
python main1.py
```