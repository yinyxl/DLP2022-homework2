# DLP2022-homework2
## 代码模块说明
1.main.py中是启动文件  

2.train.py和validation.py分别是train和validate的代码部分  

3.models.self文件夹中存放自己改动的resnet模型，resnet_18,resnet18_v1,resnet18_v2分别是原模型和两个修改后的版本，训练不同模型时使用不同文件中的resnet18函数即可     

## 代码复现说明
1.main.py中修改了一些超参数，train.py和validation.py中添加了tensorboard的代码，models.self文件夹中resnet18是原始模型，resnet18_v1中将最后的平均池化层和全连接层改为三层全连接层结合dropout；  在resnet18_v2中将avg pool 改为max pool。  

2.运行不同模型时需要注释掉__init__.py文件中不用的模型对应的文件调用，因为函数resnet18在各模型文件中都同名，不注释会导致多个同名函数。   

3.启动命令行示例：python main.py --epoch 20 --lr 0.1 -a resnet18    
