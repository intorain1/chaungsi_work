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
2. **获取文件(记得给个star哈）**
   ```bash
   git clone https://github.com/intorain1/detector.git
   
3. **运行yolo结果**
   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   pip install ultralytics
   cd ultralytics
   ```
   然后去看yolo文件夹的readme吧

4. **运行传统代码结果**
 
   懒得可视化了 效果太差）记得要修改network.py里的路径
   ```bash
   cd old
   python network.py
   ```

6. **运行我们自己搭的代码的结果**
   ```bash
   cd ours
   python gui.py
   ```
   
