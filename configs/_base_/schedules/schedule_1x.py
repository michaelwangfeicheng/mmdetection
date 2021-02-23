# optimizer
# 这里注意配置都是默认8块gpu的训练，如果用一块gpu训练，需要在lr/8
# 学习率lr和总的batch size数目成正比，例如：8卡GPU  samples_per_gpu = 2的情况（相当于总的batch size = 8*2）,学习率lr = 0.02
# 如果我是单卡GPU samples_per_gpu = 4的情况，学习率lr应该设置为:0.02*(4/16) = 0.005
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
