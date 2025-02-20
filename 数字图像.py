#引用库
import tkinter as tk  # 导入tkinter模块，别名tk
from PIL import Image, ImageTk  # 导入Image和ImageTk类
from tkinter import filedialog  # 导入filedialog模块
import cv2   # 导入cv2模块
import tkinter.messagebox   # 导入tkinter.messagebox模块
import numpy as np   # 导入numpy模块，并使用np作为别名
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，并使用plt作为别名

#定义窗口
mainWindow = tk.Tk()  # 用一个变量名来定义一个窗口
maxwidth, maxheight = mainWindow.maxsize()  # 获取屏幕的最大尺寸
width = 1500  # 定义窗口的宽
height = 800  # 定义窗口的高
hor_center = (maxwidth - width) / 2  # 水平中心
ver_center = (maxheight - height) / 2  # 垂直中心
center = '%dx%d+%d+%d' % (width, height, hor_center, ver_center)
mainWindow.geometry(center)  # 窗口位于屏幕正中心
mainWindow.title("图像处理")  # 设置mainWindow的标题为"图像处理"
mainWindow.config(bg="white")  # 设置mainWindow的背景颜色为白色




#全局变量初始化
photo=None  # photo变量，初始值为None
img=None   # img变量，初始值为None
photo1=None  # photo1变量，初始值为None
photo2=None  # photo2变量，初始值为None
img1=None   # img1变量，初始值为None
image2=None  # image2变量，初始值为None
file_path1=None  # file_path1变量，初始值为None
file_path2=None  # file_path2变量，初始值为None



#                                               定义函数
#函数功能：在文件中选择图像
def folder_Path():
    # 获取文件路径
     # 弹出文件对话框，允许选择jpg、jpeg、png格式的图片文件
    # 将选择的文件路径转换为字符串类型并赋值给file_path变量
    file_path = str(filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]))
    # 返回文件路径
    return file_path


#函数功能：向第一个Label中放置图片并显示
# 定义一个函数，用于显示输入图像
def input1_Show():
    global photo  # 声明photo为全局变量
    global img  # 声明img为全局变量
    global file_path1  # 声明file_path1为全局变量

    file_path1 = folder_Path()  # 调用folder_Path()函数获取文件夹路径
    image = cv2.imread(file_path1)  # 使用cv2.imread()函数读取图片文件，存储在image变量中
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 使用cv2.cvtColor()函数将图像从BGR颜色空间转换为RGB颜色空间
    image = cv2.resize(image, (400, 300))  # 使用cv2.resize()函数调整图像大小为400x300
    image = Image.fromarray(image)  # 使用PIL模块的Image.fromarray()方法将图像转换为Image对象
    photo = ImageTk.PhotoImage(image)  # 使用PIL模块的ImageTk.PhotoImage()方法将Image对象转换为PhotoImage对象
    imgelabel1 = tk.Label(mainWindow, image=photo)  # 创建一个Label组件，使用imgelabel1作为组件名，mainWindow作为父组件，image作为显示内容
    imgelabel1.place(x=70, y=0)  # 将组件imgelabel1放置在坐标(70, 0)处

#函数功能：向第二个Label中放置图片并显示
# 定义一个函数，用于显示输入图像
def input2_Show():
    global photo1  # 声明photo1为全局变量
    global img1   # 声明img1为全局变量
    global file_path2   # 声明file_path2为全局变量

    file_path2 = folder_Path()    # 调用folder_Path()函数获取文件夹路径
    image1 = cv2.imread(file_path2)  # 使用cv2.imread()函数读取图片文件，存储在image变量中
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # 使用cv2.cvtColor()函数将图像从BGR颜色空间转换为RGB颜色空间
    image1=cv2.resize(image1,(400,300))  # 使用cv2.resize()函数调整图像大小为400x300
    image1 = Image.fromarray(image1)   # 使用PIL模块的Image.fromarray()方法将图像转换为Image1对象
    photo1= ImageTk.PhotoImage(image1)  # 使用PIL模块的ImageTk.PhotoImage()方法将Image1对象转换为PhotoImage对象
    imgelabel2 = tk.Label(mainWindow, image=photo1) # 创建一个Label组件，使用imgelabel1作为组件名，mainWindow作为父组件，image作为显示内容
    imgelabel2.place(x=500, y=0)  # 将组件imgelabel1放置在坐标(500, 0)处


#函数功能：输出处理后的图片
def output_show(img):
    global photo2
    global image2
    image2=img   # 将输入图像赋值给image2变量
    image2=cv2.resize(img,(500,500))  # 调整图像大小为500x500像素
    # 将图像转换为Tkinter可以识别的图像格式
    image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式
    image2= Image.fromarray(image2)
    photo2 = ImageTk.PhotoImage(image2)  # 用PIL模块的PhotoImage打开
    imgelabel3 = tk.Label(mainWindow, image=photo2, width=500, height=500)
    imgelabel3.place(x=1000, y=0)

def output_show1(img):
    global photo3
    global image3
    image3=img
    image3=cv2.resize(img,(500,500))
    # 将图像转换为Tkinter可以识别的图像格式
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    image3= Image.fromarray(image3)
    photo3 = ImageTk.PhotoImage(image3)  # 用PIL模块的PhotoImage打开
    imgelabel4 = tk.Label(mainWindow, image=photo3, width=500, height=500)  # 创建一个标签，用于显示图像
    imgelabel4.place(x=500, y=0)

def output_show2(img):
    global photo4
    global image4
    image4=img
    image4=cv2.resize(img,(500,500))
    # 将图像转换为Tkinter可以识别的图像格式
    image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
    image4= Image.fromarray(image4)
    photo4 = ImageTk.PhotoImage(image4)  # 用PIL模块的PhotoImage打开
    imgelabel5 = tk.Label(mainWindow, image=photo4, width=1000, height=500)
    imgelabel5.place(x=500, y=0)
#函数功能：保存处理后的图片
def Save():
    global image2
    file_path=filedialog.asksaveasfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    image2.save(file_path)


#函数功能：实现加运算
def cv_plus():
    # 读取 RGB 图像
    global file_path1
    global file_path2
    filepath1 = file_path1
    filepath2 = file_path2
    img1 = cv2.imread(filepath1, 1)  # 读取图像1
    img2 = cv2.imread(filepath2, 1)  # 读取图像2

    X = cv2.resize(img1, (500, 500))  # 调整图像1的大小为500x500
    Y = cv2.resize(img2, (500, 500))  # 调整图像2的大小为500x500

    # 任务1：对 X,Y 进行加法运算
    result = cv2.add(X, Y)  # 对图像1和图像2进行相加操作
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # 将结果图像的通道从BGR转换为RGB
    output_show(result)  # 展示结果图像


#函数功能：实现减运算
def cv_Subtract():
    global file_path1
    global file_path2

    # 获取文件路径1和文件路径2的全局引用
    filepath1 = file_path1
    filepath2 = file_path2

    # 读取图像1
    img1 = cv2.imread(filepath1, 1)
    # 读取图像2
    img2 = cv2.imread(filepath2, 1)

    # 将图像1调整为500x500大小
    X = cv2.resize(img1, (500, 500))
    # 将图像2调整为500x500大小
    Y = cv2.resize(img2, (500, 500))
    
    # 对图像1和图像2进行相减操作
    result = cv2.subtract(X, Y)
    # 将结果由BGR格式转换为RGB格式
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # 显示结果图像
    output_show(result)


#函数功能实现与运算
def cv_And():
    global file_path1  # 全局变量，文件路径1
    global file_path2  # 全局变量，文件路径2
    filepath1 = file_path1  # 本地变量，文件路径1
    filepath2 = file_path2  # 本地变量，文件路径2
    img1 = cv2.imread(filepath1, 1)  # 读取图像1
    img2 = cv2.imread(filepath2, 1)  # 读取图像2
    X = cv2.resize(img1, (500, 500))  # 调整图像1大小为500x500
    Y = cv2.resize(img2, (500, 500))  # 调整图像2大小为500x500
    result = X & Y  # 对图像1和图像2进行按位与操作
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # 转换图像颜色空间为RGB
    output_show(result)  # 展示结果图像


#函数功能：实现或运算
def cv_Or():
    global file_path1  # 全局变量，文件路径1
    global file_path2  # 全局变量，文件路径2
    filepath1 = file_path1  # 将文件路径1赋值给局部变量filepath1
    filepath2 = file_path2  # 将文件路径2赋值给局部变量filepath2
    img1 = cv2.imread(filepath1, 1)  # 读取图片1，使用多通道模式
    img2 = cv2.imread(filepath2, 1)  # 读取图片2，使用多通道模式
    X = cv2.resize(img1, (500, 500))  # 将图片1调整为500x500大小
    Y = cv2.resize(img2, (500, 500))  # 将图片2调整为500x500大小
    result=X|Y  # 对图片1和图片2进行按位或操作
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # 将结果图片的色彩空间转换为RGB
    output_show(result)  # 展示结果图片


#函数功能：实现图像灰度化运算
def convert_to_gray():
    global file_path1
    global file_path2
    
    # 判断file_path1和file_path2是否均不为空
    if file_path2 != None and file_path1 != None:
        file_path2 = None
    else:
        filepath1 = file_path1
        image = cv2.imread(filepath1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output_show(image)


#函数功能：实现图像二值化
def hdss_su():
    global file_path1
    img = cv2.imread(file_path1)
    # 任务1先将img转为灰度
    # 再对其使用两个阈值方法进行二值化，阈值为127
    ###########Begin###########
    Image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将img转为灰度图像
    ret, BINARY = cv2.threshold(Image, 127, 255, cv2.THRESH_BINARY)  # 使用二值化方法将灰度图像转为二值图像，阈值为127
    ###########End#############
    # 返回值为Image, BINARY, BINARY_INV
    result = BINARY
    output_show(result)



#函数功能：对数变换
def change():
    global file_path1
    # 读取图像
    #对数变换
    X = cv2.imread(file_path1, 0)   # 读取文件路径为file_path1的图像，并转换为灰度图像
    C = 255 / np.log(1 + 255)  # 计算对数变换的缩放因子
    ##########  End  ##########

    # 任务2：对 X 进行对数变换
    ########## Begin ##########
    result = C * np.log(1 + X)  #对 X 进行对数变换
    ##########  End  ##########

    # 由于 np.log() 函数会降低精度，所以将结果转换成高精度
    result = np.array(result, np.float64)   # 将结果转换为双精度浮点数类型
    output_show(result)  # 调用输出显示函数，显示变换后的图像



#函数功能：实现图像的幂次变换
def power_Operation():
    global file_path1
    # 读取图像
    image = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)

    # 将图像转换为浮点数，以便进行幂次运算
    image_float = image.astype(np.float32)

    # 定义幂次数
    power = 2.0

    # 对图像进行幂次运算
    result = cv2.pow(image_float, power)
    result_normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    output_show(result_normalized)


#函数功能：实现线性变换
def line_Change():
    global file_path1
    X = cv2.imread(file_path1, 0)

    # 任务1：定义区间
    ########## Begin ##########
    a = 20
    b = 241
    c = 0
    d = 255
    ##########  End  ##########

    # 任务2：对 X 进行线性变换
    ########## Begin ##########
    result = (d - c) / (b - a) * X + (b * c - a * d) / (b - a)
    output_show(result)



#函数功能：实现图片旋转
def rotate_image(event):
    global file_path1
    # 获取滑动条的值
    degree = slider.get()

    # 读取图像
    img = cv2.imread(file_path1)

    # 获取图像的尺寸
    (h, w) = img.shape[:2]

    # 计算旋转中心
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1)

    # 应用旋转矩阵
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))
    rotated_img=cv2.cvtColor(rotated_img,cv2.COLOR_BGR2RGB)
    output_show(rotated_img)


#函数功能：实现图片镜像变换
def img_Flip_horizontally():
    global file_path1
    img = cv2.imread(file_path1)

    # 沿着垂直方向翻转图像
    img_flip_vertically = cv2.flip(img, 0)
    img_flip_vertically=cv2.cvtColor(img_flip_vertically,cv2.COLOR_BGR2RGB)
    output_show(img_flip_vertically)

    # 沿着水平方向翻转图像
    img_flip_horizontally = cv2.flip(img, 1)
    output_show1(img_flip_horizontally)


#函数功能：实现图片透视变换
def dst_Img():
    global file_path1
    # 读取图像
    img = cv2.imread(file_path1)

    # 定义源图像中的四个点
    src_pts = np.float32([[0, 0], [500,0 ], [500, 500], [0, 500]])

    # 定义目标图像中的四个点
    dst_pts = np.float32([[100, 100], [400, 100], [400, 400], [100, 400]])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 进行透视变换
    dst_img = cv2.warpPerspective(img, M, (300, 300))
    output_show(dst_img)



#函数功能：实现图像的直方图均衡化，并将处理前后的图像放在同一张图上
def histogram_equalization():
    global file_path1
    img=cv2.imread(file_path1)
    # 检查图像是否为灰度图像
    if img.shape[2] != 1:
        # 如果不是，转换为灰度图像
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 然后进行直方图均衡化
        equalized_img = cv2.equalizeHist(gray_img)
    else:
        # 如果是灰度图像，直接进行直方图均衡化
        equalized_img = cv2.equalizeHist(img)
    equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)
    equalized_img = cv2.resize(equalized_img,(250,500))

    # 获取原图的大小并修改为(250, 500)
    resized_img = cv2.resize(img, (250, 500))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # 拼接图像
    equalized_img = np.array(equalized_img)
    resized_img = np.array(resized_img)
    combined = cv2.vconcat([equalized_img, resized_img])
    output_show(combined)


#函数功能：实现图像的采样与量化
def quantized_Img():
    global file_path1
    # 读取图像
    img = cv2.imread(file_path1)

    # 图像量化
    # cv2.quantize()函数接受以下参数：原图像，量化等级，颜色数目，颜色量化等级，转换类型
    # 例如，这里我们将图像量化到8种颜色，颜色等级为256，转换类型为CV_8UC1，即8位无符号单通道图像
    quantized_img = np.uint8(img)
    quantized_img=cv2.cvtColor(quantized_img, cv2.COLOR_BGR2RGB)
    equalized_img = cv2.resize(quantized_img, (250, 500))

    # 获取原图的大小并修改为(250,500)
    resized_img = cv2.resize(img, (250, 500))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # 拼接图像
    equalized_img = np.array(equalized_img)
    resized_img = np.array(resized_img)
    combined = cv2.hconcat([equalized_img, resized_img])
    output_show(combined)


#函数功能：实现人体骨骼在空间中增强
def img_change():
    

    global file_path1

    # 声明全局变量file_path1

    image = cv2.imread(file_path1)  # 使用cv2库的imread函数读取指定路径的图像，并保存到变量image中

    image = cv2.resize(image, (250, 500))  # 使用cv2库的resize函数将图像大小调整为250x500，并保存到变量image中

    # 将图像转换为YUV颜色空间
    image_yuv = cv2.resize(image, (250, 500))  # 使用cv2库的resize函数将图像大小调整为250x500，并保存到变量image_yuv中，此处是为了后续转换为YUV颜色空间做准备

    image_yuv = cv2.cvtColor(image_yuv, cv2.COLOR_BGR2YUV)  # 使用cv2库的cvtColor函数将变量image_yuv的图像从BGR颜色空间转换为YUV颜色空间

    image_yuv = np.array(image_yuv)  # 将图像保存到NumPy数组中

    # 对Y通道进行直方图均衡化
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])  # 对image_yuv的Y通道进行直方图均衡化，使图像对比度增强

    # 将图像转换回BGR颜色空间
    enhanced_image = cv2.resize(image_yuv, (250, 500))  # 使用cv2库的resize函数将图像大小调整为250x500，并保存到变量enhanced_image中，此处是为了后续转换为BGR颜色空间做准备

    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_YUV2BGR)  # 使用cv2库的cvtColor函数将变量enhanced_image的图像从YUV颜色空间转换为BGR颜色空间

    enhanced_image = np.array(enhanced_image)  # 将图像保存到NumPy数组中

    # 显示处理后的图像
    enhanced_image1 = cv2.resize(enhanced_image, (250, 500))  # 使用cv2库的resize函数将图像大小调整为250x500，并保存到变量enhanced_image1中

    enhanced_image1 = cv2.cvtColor(enhanced_image1, cv2.COLOR_BGR2RGB)  # 使用cv2库的cvtColor函数将变量enhanced_image1的图像从BGR颜色空间转换为RGB颜色空间，以便在屏幕上正确显示

    enhanced_image1 = np.array(enhanced_image1)  # 将图像保存到NumPy数组中

    combined1 = cv2.hconcat([image, image_yuv])  # 使用cv2库的hconcat函数将变量image和image_yuv的图像水平合并，并保存到变量combined1中

    combined2 = cv2.hconcat([enhanced_image, enhanced_image1])  # 使用cv2库的hconcat函数将变量enhanced_image和enhanced_image1的图像水平合并，并保存到变量combined2中

    output_show1(combined1)  # 调用函数output_show1显示变量combined1的图像

    output_show(combined2)  # 调用函数output_show显示变量combined2的图像
#函数功能：实现图片膨胀
def img_Dilation():
    global file_path1
    img = cv2.imread(file_path1, 0)  # 读取灰度图像

    # 定义结构元素（例如，3x3矩阵）
    kernel = np.ones((3, 3), np.uint8)  # 使用全1矩阵作为结构元素

    # 使用膨胀函数
    dilation = cv2.dilate(img, kernel, iterations=1)   # 对图像进行膨胀操作
    output_show(dilation)


#函数功能：实现图像的傅里叶变换
def img_dft():
    global file_path1
    # 读取图像
    img = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)

    # 对图像进行傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 取对数，将幅度变换到可显示的范围内
    log_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    output_show(log_spectrum)


#函数功能：实现图像腐蚀
def erosion_Image():
    global file_path1  # 定义全局变量file_path1
    # 读取图像
    img = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)  # 使用cv2.imread函数读取图像，图像以灰度图方式读取

    # 定义结构元素，这里我们使用3x3的方形
    kernel = np.ones((3, 3), np.uint8)  # 使用np.ones函数创建一个3x3的方形矩阵，元素全为1

    # 使用腐蚀操作
    erosion = cv2.erode(img, kernel, iterations=1)  # 使用cv2.erode函数进行腐蚀操作，对img进行一次腐蚀操作，使用kernel作为结构元素
    output_show(erosion)  # 调用output_show函数显示结果图像
  

#函数功能：实现图片开运算
def img_Opening():
    global file_path1
    img = cv2.imread(file_path1, 0)  # 读取图像文件为灰度图像

    # 定义结构元素（例如，3x3矩阵）
    kernel = np.ones((3, 3), np.uint8)

    # 使用开运算函数
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算，用于去除小的噪点
    output_show(opening)  # 调用输出显示函数显示开运算结果


#函数功能：实现图片闭运算
def img_Closing():
    global file_path1
    # 读取图像
    img = cv2.imread(file_path1, 0)

    # 定义结构元素（例如，3x3矩阵）
    kernel = np.ones((3, 3), np.uint8)

    # 使用闭运算函数
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    output_show(closing)


#函数功能：实现图像的边缘检测
def img_Canny():
    global file_path1
    # 读取图像
    img = cv2.imread(file_path1, 0)

    # 使用Canny边缘检测算法
    edges = cv2.Canny(img, 100, 200)
    output_show(edges)


#函数功能：实现图像的hough变换
def hough_transform():
    global file_path1
    # 读取图像
    img = cv2.imread(file_path1)
    if img is None:
        print("Image is None, please check the file path.")
        return

        # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)
    if edges is None:
        print("Edges is None, please check the Canny edge detection.")
        return

        # 使用Hough变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        print("Lines is None, please check the Hough transform.")
        return

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_show(img)


#函数功能：实现检测图像中硬币的个数
def denoise_image():
    global file_path1
    img=cv2.imread(file_path1)  # 读取图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度化
    denoised_img = cv2.GaussianBlur(gray_img, (3, 3), 0)#降噪
    ret, binary_img = cv2.threshold(denoised_img, 127, 255, cv2.THRESH_BINARY) #二值化
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义一个结构元素
    closing_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)  # 对图像进行闭运算操作
    eroded_img = cv2.erode(closing_img, kernel)  # 对图像进行腐蚀操作
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # 寻找轮廓
    coin_count = 0
    for contour in contours:  # 遍历每个轮廓
        area = cv2.contourArea(contour)  # 计算轮廓面积
        if 600 < area< 1740 :   # 判断轮廓面积是否在指定范围内
            coin_count += 1  # 硬币数量加1
    result="硬币数量为: " + str(coin_count)   # 构建结果显示信息
    tk.messagebox.showinfo("结果",result)   # 显示结果显示信息

def Empty():
    global file_path1  # 定义全局变量file_path1
    global file_path2  # 定义全局变量file_path2
    
    if file_path1 != None and file_path2 != None:  # 判断file_path1和file_path2是否都不为None
        file_path1 = None  # 将file_path1赋值为None
        output_show(cv2.imread(file_path1))  # 调用output_show函数显示通过cv2.imread函数读取的file_path1对应的图像
        file_path2 = None  # 将file_path2赋值为None
        output_show1(cv2.imread(file_path2))  # 调用output_show1函数显示通过cv2.imread函数读取的file_path2对应的图像

#定义菜单栏
menu_bar = tk.Menu(mainWindow)  # 创建菜单栏对象
file_menu = tk.Menu(menu_bar, tearoff=0)  # 创建文件菜单对象
menu_bar.add_cascade(label="文件", menu=file_menu)  # 将文件菜单添加到菜单栏中
file_menu.add_command(label="退出", command=mainWindogw.quit)  # 添加退出命令到文件菜单中
file_menu.add_command(label="清空", command=Empty())  # 添加清空命令到文件菜单中
file_menu.add_command(label="保存", command=Save)  # 添加保存命令到文件菜单中
file_menu1 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu1
menu_bar.add_cascade(label="代数逻辑运算", menu=file_menu1)  # 向menu_bar添加一个下拉菜单，标签为"代数逻辑运算"，菜单为file_menu1
file_menu1.add_command(label="加运算", command=lambda:cv_plus())  # 向file_menu1菜单添加一个命令，标签为"加运算"，命令为调用cv_plus()函数
file_menu1.add_command(label="减运算", command=lambda:cv_Subtract())  # 向file_menu1菜单添加一个命令，标签为"减运算"，命令为调用cv_Subtract()函数
file_menu1.add_command(label="与运算", command=lambda:cv_And())  # 向file_menu1菜单添加一个命令，标签为"与运算"，命令为调用cv_And()函数
file_menu1.add_command(label="或运算", command=lambda:cv_Or())  # 向file_menu1菜单添加一个命令，标签为"或运算"，命令为调用cv_Or()函数

# 创建一个文件菜单2
file_menu2 = tk.Menu(menu_bar, tearoff=0)
# 向菜单栏添加一个名为"图像灰度化"的下拉菜单，并将文件菜单2设置为其选项
menu_bar.add_cascade(label="图像灰度化", menu=file_menu2)
# 向下拉菜单中添加一个命令类型的选项，显示为"图像灰度化"，并设置其命令为一个lambda函数
file_menu2.add_command(label="图像灰度化", command=lambda:convert_to_gray())

# 创建一个文件菜单file_menu3，设置tearoff属性为0，表示不提供拖出菜单的功能
file_menu3 = tk.Menu(menu_bar, tearoff=0)
# 向菜单栏menu_bar中添加一个名为"图像二值化"的下拉菜单file_menu3
menu_bar.add_cascade(label="图像二值化", menu=file_menu3)
# 在file_menu3菜单中添加一个命令类型选项，显示为"图像二值化"，点击后执行hdss_su()函数
file_menu3.add_command(label="图像二值化", command=lambda:hdss_su())

file_menu4 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu4
menu_bar.add_cascade(label="灰度变换", menu=file_menu4)  # 将菜单对象file_menu4添加到menu_bar中并设置标签为"灰度变换"
file_menu4.add_command(label="对数变换", command=lambda: change())  # 在菜单file_menu4中添加一个选项"对数变换"，点击选项会调用change()函数
file_menu4.add_command(label="幂次变换", command=lambda: power_Operation())  # 在菜单file_menu4中添加一个选项"幂次变换"，点击选项会调用power_Operation()函数
file_menu4.add_command(label="线性变换", command=lambda: line_Change())  # 在菜单file_menu4中添加一个选项"线性变换"，点击选项会调用line_Change()函数

file_menu5 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu5
menu_bar.add_cascade(label="几何变换", menu=file_menu5)  # 向menu_bar菜单中添加一个子菜单，标签为"几何变换"，指向file_menu5
file_menu5.add_command(label="镜像变换", command=lambda:img_Flip_horizontally())  # 向file_menu5菜单中添加一个选项，标签为"镜像变换"，点击时执行img_Flip_horizontally()函数
file_menu5.add_command(label="透视变换", command=lambda:dst_Img())  # 向file_menu5菜单中添加一个选项，标签为"透视变换"，点击时执行dst_Img()函数

file_menu6 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu6，指定其为menu_bar的子菜单
menu_bar.add_cascade(label="直方图均衡化", menu=file_menu6)  # 将label为"直方图均衡化"的菜单添加到menu_bar中，并指定其为file_menu6的子菜单
file_menu6.add_command(label="直方图均衡化", command=lambda:histogram_equalization())  # 在file_menu6中添加一个选项label为"直方图均衡化"的菜单项，并指定其点击后的命令为调用函数histogram_equalization()

file_menu7 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu7，用于添加图像的采样和量化的子菜单
menu_bar.add_cascade(label="图像的采样和量化", menu=file_menu7)  # 将file_menu7作为子菜单添加到menu_bar菜单中，并设置显示的标签为"图像的采样和量化"
file_menu7.add_command(label="图像的采样和量化", command=lambda:quantized_Img())  # 在file_menu7菜单中添加一个选项，显示的标签为"图像的采样和量化"，点击该选项后执行quantized_Img()函数

file_menu8 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu8，继承自tk.Menu类，并设置tearoff属性为False
menu_bar.add_cascade(label="人体骨骼", menu=file_menu8)  # 将label为"人体骨骼"的菜单file_menu8添加到menu_bar中
file_menu8.add_command(label="人体骨骼", command=lambda:img_change())  # 在菜单file_menu8中添加一个label为"人体骨骼"的命令，命令是调用img_change()函数

file_menu9 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu9，指定其父对象为menu_bar，并设置tearoff属性为False
menu_bar.add_cascade(label="傅里叶变换", menu=file_menu9)  # 将菜单对象file_menu9添加到menu_bar中，并设置其标签为"傅里叶变换"
file_menu9.add_command(label="傅里叶变换", command=lambda:img_dft())  # 在菜单对象file_menu9中添加一个命令，标签为"傅里叶变换"，并设置其命令为调用img_dft()函数

file_menu10 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu10
menu_bar.add_cascade(label="形态学处理", menu=file_menu10)  # 将菜单对象file_menu10添加到menu_bar，设置显示文本为"形态学处理"
file_menu10.add_command(label="腐蚀", command=lambda: erosion_Image())  # 在file_menu10菜单中添加一个选项"腐蚀"，点击时执行函数erosion_Image()
file_menu10.add_command(label="膨胀", command=lambda: img_Dilation())  # 在file_menu10菜单中添加一个选项"膨胀"，点击时执行函数img_Dilation()
file_menu10.add_command(label="开运算", command=lambda: img_Opening())  # 在file_menu10菜单中添加一个选项"开运算"，点击时执行函数img_Opening()
file_menu10.add_command(label="闭运算", command=lambda: img_Closing())  # 在file_menu10菜单中添加一个选项"闭运算"，点击时执行函数img_Closing()

file_menu11 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu11
menu_bar.add_cascade(label="边缘检测和图像分割", menu=file_menu11)  # 将菜单file_menu11添加到menu_bar中，并设置标签为"边缘检测和图像分割"
file_menu11.add_command(label="边缘检测", command=lambda:img_Canny())  # 在file_menu11中添加一个选项"边缘检测"，并设置其命令为调用img_Canny()函数
file_menu11.add_command(label="hough变换", command=lambda:hough_transform())  # 在file_menu11中添加一个选项"hough变换"，并设置其命令为调用hough_transform()函数

file_menu12 = tk.Menu(menu_bar, tearoff=0)  # 创建一个菜单对象file_menu12，用于文件菜单
menu_bar.add_cascade(label="硬币检测技术", menu=file_menu12)  # 将硬币检测技术添加到菜单栏menu_bar中的下拉菜单中
file_menu12.add_command(label="硬币检测", command=lambda:denoise_image())  # 在下拉菜单file_menu12中添加一个选项硬币检测，点击该选项时执行denoise_image()函数

mainWindow.config(menu=menu_bar)  # 将菜单栏添加到窗口实例


#定义滑动条
# 创建一个垂直方向上的滑块
slider = tk.Scale(mainWindow, from_=0, to=360, orient=tk.VERTICAL, resolution=1, tickinterval=10, length=720)
# 指定滑块在窗口中的位置
slider.place(x=0, y=0)
# 配置滑块命令，用于旋转图像
slider.config(command=rotate_image)


#定义按钮
input1 = tk.Button(mainWindow, text="选择图像1", width=8, height=1, command=lambda:input1_Show(),fg="black")  # 创建一个按钮，用于选择图像1
input1.configure(background="white")  # 设置按钮的背景颜色为白色
input1.place(x=250, y=550)  # 设置按钮的位置为(250, 550)
input2 = tk.Button(mainWindow, text="选择图像2", width=8, height=1, command=lambda:input2_Show(),fg="black")  # 创建一个按钮，用于选择图像2
input2.configure(background="white")  # 设置按钮的背景颜色为白色
input2.place(x=750, y=550)  # 设置按钮的位置为(750, 550)
mainWindow.mainloop()  # 运行主窗口的事件循环
