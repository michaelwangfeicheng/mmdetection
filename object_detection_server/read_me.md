# mmdetection_api.py 使用

```python
import os
import cv2 as cv
from object_detection_server.mmdetection_api import seal_detection
print(f"开始测试 mmdetection_api")
# 加载模型
from object_detection_server.load_model import model
from object_detection_server.config.configs import score_thr
# 使用 CV 读取图片 :BGR格式
test_data_dir = os.path.join('object_detection_server', 'test_data', 'seal_data_real','test')
test_data_filenames = os.listdir(test_data_dir)
image_index = 1
test_img_path = os.path.join(test_data_dir, test_data_filenames[image_index])
test_img_bgr = cv.imread(test_img_path)
# 使用模型进行预测
result,processed_image = seal_detection(model=model,image=test_img_bgr,score_thr=score_thr)
print(f"测试 mmdetection_api: result={result}")
```

## 配置参数 object_detection_server/config/configs.py
- 模型加载设备
使用CPU: device= 'cpu' 
使用GPU: device = 'cuda:0'
- 模型配置文件
config_file = 'object_detection_server/model/model_6_classes/faster_rcnn_r50_fpn_1x_003.txt'
- 模型 checkpoint
checkpoint_file = 'object_detection_server/model/model_6_classes/epoch_12.pth'
- 目标识别阈值
score_thr = 0.3
- api识别结果是否返回处理过的图片
if_ger_processed_image = False

## 模型
- 位置： object_detection_server/model/model_6_classes/epoch_12.pth
- 模型配置：object_detection_server/model/model_6_classes/faster_rcnn_r50_fpn_1x_003.txt



