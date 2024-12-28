## 运行方式
1. **文件路径**
   
   请将yolo文件夹中文件放置到ultralytics目录下
   
   project-name/
│
├── dir1/
│   ├── file1.ext
│   └── file2.ext
│
├── dir2/
│   ├── sub-dir1/
│   │   └── file3.ext
│   └── sub-dir2/
│
└── dir3/
    ├── file4.ext
    └── file5.ext

3. **运行**
   
   ```bash
   python gui.py
5. **训练**
   
     请参照data.yaml格式配置数据集，注意，数据集要符合yolo官方标准，可以在ultralytics的readme中获取
     更换train.py中的路径，相信大家会看得懂的
