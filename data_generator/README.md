# Creating Synthetic Image Datasets
This tool helps create synthetic data for object detection modeling. Given
a folder of background images and object images, this tool iterates through each
background and superimposes objects within the frame in random locations,
automatically annotating as it goes. The tool also resizes the icons to help the
model generalize better to the real world.


## 人工数据生成
Run the `synthetic_data_generator.py` script to generate hundreds/thousands of synthetic training
images for object detection models.

Output images will be placed in the `data/` subfolder once done.
- backgrounds： 存放不同的图片
- objects： 按照 label作为目录 存放不同的图片，生成的时候 label 使用该目录
- annotations: 生成标注的 annotations train.json + dev.json
- train/dev/test:将生成的图片分布放入各自目录

### Args


## 人工合成数据验证
Run `test_synthetic_data.ipynb`