# XNet
Work done recently for salient object detection



1. 按照`Google Styleguide for Python`代码风格重构了显著性目标检测的`torch_utils.py`、`train.py`、和`test.py`；

2. **Bug Fixes：**

```
1.init_seeds(2 + opt.batch_size)
2.使用model_info(model, verbose=True)代替logging.info(summary(model, (3, 320, 320)))
4.添加tqdm和pbar显示训练进度条
5.解决训练时显存抖动
7.使用argparse.ArgumentParser()
8.time_synchronized()
9.训练时显示已用时间，测试时显示数据集名称、FPS、保存路径；
10.optimizer.zero_grad()
11.使用AdamW + OneCycleLR
12. 梯度累积accumulation_steps = 
```

