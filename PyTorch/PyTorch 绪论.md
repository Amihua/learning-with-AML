@[TOC](Task1 ： PyTorch 绪论)
## 深度学习 PyTorch or TensorFlow？

![请添加图片描述](https://img-blog.csdnimg.cn/a866fdf467d643428f918369427f2b61.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQW1paHVhIExhdQ==,size_19,color_FFFFFF,t_70,g_se,x_16)


> PyTorch更有利于研究人员、爱好者、小规模项目等快速搞出原型。而TensorFlow更适合大规模部署，特别是需要跨平台和嵌入式部署时。[1](https://zhuanlan.zhihu.com/p/28636490).

## 什么是Torch ？又什么是PyTorch
[2](https://github.com/zergtant/pytorch-handbook/blob/master/chapter1/1.1-pytorch-introduction.md).
### Torch是一个与Numpy类似的张量（Tensor）操作库

	与Numpy不同的是Torch对GPU支持的很好，Lua是Torch的上层包装。

### PyTorch和Torch使用包含所有相同性能的C库
	TH, THC, THNN, THCUNN它们将继续共享这些库。PyTorch和Torch都使用的是相同的底层，只是使用了不同的上层包装语言。

### PyTorch是一个基于Torch的Python开源机器学习库

	用于自然语言处理等应用程序。 它主要由Facebook的人工智能研究小组开发。Uber的"Pyro"也是使用的这个库。

### PyTorch是一个Python包，提供两个高级功能：

	- 具有强大的GPU加速的张量计算（如NumPy）
	- 包含自动求导系统的的深度神经网络

## 安装PyTorch

1. CUDA版本:**NVIDIA 控制面板 => 系统信息 => 组件** 
![请添加图片描述](https://img-blog.csdnimg.cn/e44a387685724dc28cb4026f2386a725.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQW1paHVhIExhdQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
2. 从pytorch官网获取安装命令 https://pytorch.org/

![请添加图片描述](https://img-blog.csdnimg.cn/ba1b288917fa45e3b7b06ca6585a986f.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQW1paHVhIExhdQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
3. 验证安装是否成功

   ```python
   import torch 
   torch.cuda.is_available()
   ```
![请添加图片描述](https://img-blog.csdnimg.cn/a1b2eef9b1924f1f99cb8bfb35290318.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQW1paHVhIExhdQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
   ## Hello PyTorch

   ```python
   import torch
   x = torch.rand(5, 3)
   print(x)
   ```
![请添加图片描述](https://img-blog.csdnimg.cn/ccd372492b3c4abfae96f25b43341c31.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAQW1paHVhIExhdQ==,size_20,color_FFFFFF,t_70,g_se,x_16)




