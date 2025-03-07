# 图像处理程序

这是一个基于Python的图像处理程序，提供了丰富的图像处理功能。该程序使用tkinter构建图形用户界面，支持多种图像处理操作。

## 功能特点

### 基础操作
- 支持打开和保存图像文件
- 支持同时处理两张图像
- 提供图像清空功能

### 代数逻辑运算
- 图像加法运算
- 图像减法运算
- 图像与运算
- 图像或运算

### 图像变换
- 灰度化处理
- 二值化处理
- 对数变换
- 幂次变换
- 线性变换

### 几何变换
- 图像旋转（带滑动条控制）
- 镜像变换
- 透视变换

### 图像增强
- 直方图均衡化
- 图像采样和量化
- 人体骨骼增强

### 频域处理
- 傅里叶变换

### 形态学处理
- 图像腐蚀
- 图像膨胀
- 开运算
- 闭运算

### 特征提取
- 边缘检测
- Hough变换
- 硬币检测技术

## 环境要求

- Python 3.x
- OpenCV (cv2)
- NumPy
- PIL (Python Imaging Library)
- Tkinter
- Matplotlib

## 安装依赖
pip install opencv-python
pip install numpy
pip install pillow
pip install matplotlib

## 使用说明

1. 运行程序后，会出现主界面窗口
2. 使用"选择图像1"和"选择图像2"按钮可以导入待处理的图像
3. 通过顶部菜单栏选择需要的图像处理功能
4. 左侧滑动条可以控制图像旋转角度
5. 处理后的图像可以通过"保存"功能保存到本地

## 界面说明

- 左侧：旋转控制滑动条（0-360度）
- 中央：图像显示区域
- 底部：图像选择按钮
- 顶部：功能菜单栏

## 注意事项

- 支持的图像格式：JPG、JPEG、PNG
- 部分功能需要同时载入两张图像
- 图像处理结果会实时显示在界面上
- 建议使用分辨率较高的显示器以获得最佳显示效果