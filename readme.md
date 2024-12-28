# 项目名称 不知道做了什么的项目

![项目图片](11.png) 
皮卡丘镇楼

嗯其实是一个检测项目，在该项目里我们分别利用yolov10和自定义网络实现了螺钉，螺母和硬币的检测？并且实现了一下可视化咳）（也许）（误）
针对mmcv的屎山再次发出吐槽，本作者立志写出一个简洁易懂的readme和运行给大家。

## 运行方式
1. **配置环境**

   ```bash
   conda create -n intorains python=3.9
   conda activate intorains
   pip install PySide6

2. **运行yolo结果（可以都试试看）**
   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   pip install ultralytics
   git clone https://github.com/intorain1/detector.git