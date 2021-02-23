# 印章识别
# 训练 ： tools/train.py

# 预测：

# 评估： 

 ```python
# single-gpu testing
CONFIG_FILE='configs/faster_rcnn_r50_fpn_1x.py'
CHECKPOINT_FILE = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
RESULT_FILE = ''
EVAL_METRICS = ''
show = False
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```
可选择的参数有： 
- RESULT_FILE：如果指定这个参数那么输出的结果将会保存在一个pkl文件中，如果不指定结果就不会保存 
- EVAL_METRICS： 这个参数依赖于数据集，对于COCO，proposal_fast, proposal, bbox, segm，
对于VOC数据集，mAP, recall， 对于cityspace， 除了支持所有的COCOmetrics还有，cityscapes * 
- show：如果指定这个参数，检测的结果就会显示在一个窗口中，这个设置只在单gpu测试时可用，同时确保GUI可用，否则可能会遇到这个错误 cannot connect to X server
在检测整个数据集时，不要指定--show这个参数


