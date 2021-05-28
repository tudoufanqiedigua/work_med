# work_med
## AI中药材识别

### 赛题背景和要求

本次比赛主题为基于中药材图片数据集的图像识别，选取了5类中药材图片，每类分别150张训练集图片以及30张测试集图片。图像分类是根据不同类别的目标在图像信息中所反映的不同特征，将其区分开来的图像处理方法。

AI中药材识别的主要任务是对已有多类中药材图片进行训练并预测，返回中药材的名称。

### 文件目录

```bash
work_med
├─ data_read.py # 数据读取
├─ data_split.py # 数据切分
├─ images  # 图片文件
├─ models # 模型
├─ window.py # UI界面
├─ test_model.py # 模型测试
├─ train_model.py # 模型训练
└─ README.md 
```

### 使用

需求安装：tensorflow-cpu == 2.3.0、pyqt5、pillow、opencv-python、matplotlib

训练模型：

```
python train_model.py
```

测试模型的准确率：

```
python test_model.py
```

图形化的界面：

```
python window.py
```

### 执行效果

#### 图形化界面

![image](https://user-images.githubusercontent.com/53803390/119920669-4ac40e00-bf9f-11eb-9e8b-f418fda6bd0d.png)

#### 界面下测试

![image](https://user-images.githubusercontent.com/53803390/119920724-60d1ce80-bf9f-11eb-903a-62147bec57cd.png)
![image](https://user-images.githubusercontent.com/53803390/119920743-66c7af80-bf9f-11eb-9cb4-bc177fde5ee3.png)
![image](https://user-images.githubusercontent.com/53803390/119920760-6d562700-bf9f-11eb-92c8-997620cc7f8f.png)
![image](https://user-images.githubusercontent.com/53803390/119920770-71824480-bf9f-11eb-992f-44f03364c3d1.png)




