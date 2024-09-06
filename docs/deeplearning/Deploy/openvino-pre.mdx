---
title: 一文搞定VisualStudio配置OpenVINO与OpenCV
authors: [Asthestarsfalll]
tags: [BaseLearn]
description: 搞一下openvino
hide_table_of_contents: false
---

Windows 下 OpenVINO 以及 OpenCV 环境配置。

## 安装 OpenVINO

在安装 OpenVINO 前，确保本地 Python 版本为 3.6-3.8 之间，Cmake 版本为 3.17 及以上，相关的安装教程网络上有很多。

在准备好前置条件之后，让我们先来到 OpenVINO Dev Tools 的 [下载地址](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)，下载选择如下：

<img src="images/2022/04/06/image-20220406211207545.png" alt="image-20220406211207545" style= {{zoom:"50%"}} />

根据需要选择版本，最好选择离线下载，在线下载可能会卡住。

下载到本地后双击打开按照提示直接安装即可。

来到选定的安装目录，默认在 `C:\Program Files (x86)\Intel` 下，这里有两个文件夹，`openvino_2021` 和 `openvino_2021.4.752`（或者是你安装的版本号，下文都以 openvino_2021.4.752 来代替）。

进入 `openvino_2021.4.752\bin`，使用 cmd 或者 terminal 运行 `setupvars.bat`，出现“[setupvars.bat] OpenVINO environment initialized”即表示配置环境变量完毕。

进入 `openvino_2021.4.752\deployment_tools\demo`，使用 cmd 或者 terminal 运行 `demo_security_barrier_camera.bat` 来进行验证安装是否成功。刚开始会下载相关的依赖，请耐心等待一会。运行成功会出现如下图片：

<img src="/images/2022/04/06/image-20220406211946858.png" alt="image-20220406211946858" style={{zoom:"50%"}} />

## Visual Studio 安装

进入 Visual Studio 官网，安装 2017 或者 2019 版本。工作复核选择如下：

<img src="/images/2022/04/06/image-20220406212142150.png" alt="image-20220406212142150" style={{zoom:"50%"}} />

语言选择英文和中文即可。

这里建议将能放在其他盘的东西都放在其他盘，不然所剩无多的 C 盘又要遭受剥削。

## Visual Studio 配置 Openvino

打开 Visual Studio，点击左上角 `视图-其他窗口-属性管理器`，会在右边得到这四个目录：

![image-20220406212411426](/images/2022/04/06/image-20220406212411426.png)

我们选择 Debug|x64 和 Release|x64，鼠标右键添加新项目属性表，在左边的通用属性中点击 VC++ 目录：

![image-20220406212537324](/images/2022/04/06/image-20220406212537324.png)

在右边常规下选中 `包含目录`，点击右侧的符号，再点击编辑即可：

![image-20220406212724376](/images/2022/04/06/image-20220406212724376.png)

包含的英文即为 `include`，接下来我们要添加三个文件夹的绝对路径：

![image-20220406212830825](/images/2022/04/06/image-20220406212830825.png)

完成后返回，点击库目录，库的英文是 library，缩写为 lib，这里我们同样添加三个目录：

![image-20220406212912849](/images/2022/04/06/image-20220406212912849.png)

**注意：第二个目录最后的文件夹取决于你当前配置的属性表所属的方案配置器——及 Release 和 Debug**

最后点击链接器 - 输入：

![image-20220406213058608](/images/2022/04/06/image-20220406213058608.png)

右边进入 `附加依赖库`，接下来要添加很多文件，首先我们进入 `openvino_2021.4.752\opencv\lib` 这个文件，里面有许多 `.lib` 文件，这里只需添加文件的名字，而非绝对路径。

方便起见可以使用脚本来得到文件名：

```python
import os
filename = 'xxx/xxx/xx/xx'
for i in os.listdir(filename):
    if not i.endswith('.lib') and i[-5] != 'd': # 当为Debug配置时，需要将不等于改为等于。
        print(i)
```

接下来再添加 `openvino_2021.4.752\deployment_tools\inference_engine\lib\intel64\Release` 下的。lib 文件即可

最后再同样配置 Debug 的属性表即可，需要注意的是，.lib 文件的文件名的最后一个字母应该是 d。

## 环境变量

我的电脑右键，属性，高级系统设置，环境变量，PATH 中配置如下变量：

![image-20220406214144949](/images/2022/04/06/image-20220406214144949.png)

![image-20220406214154996](/images/2022/04/06/image-20220406214154996.png)

## 验证

重启 Visual Studio，写一段程序来验证配置是否成功，代码如下：

```c++
#include <opencv2/opencv.hpp>
using namespace cv;
int main() {
	string filepath = "xxx//xx//xx//xxx.png";
	Mat img;
	img = imread(filepath);
	imshow("test", img);
	waitKey(0);
	return 0;
}
```

运行后，如果显示出图片及说明 opencv 的环境成功了。

需要注意的是，运行时应该将左上角的 `win32` 改为 `x64`：

![image-20220406215417154](/images/2022/04/06/image-20220406215417154.png)

如果提示找不到 `xxxx.dll`，可以在 `openvino_2021.4.752\opencv\bin` 目录下找到对应的 `.dll` 文件（若为 Release 则需要文件名最后不需要 d），将文件复制到项目所在目录的 Release 或者 Debug 下即可。
