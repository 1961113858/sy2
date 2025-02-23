# 3D-FRONT 室内场景生成

本项目使用3D-FRONT数据集训练室内场景生成模型。

## 1. 环境配置

### 1.1 创建环境
```bash
conda create -n scene_synthesis python=3.8
conda activate scene_synthesis
```

### 1.2 安装依赖
```bash
pip install torch torchvision
pip install pytorch-lightning
pip install hydra-core
pip install wandb
pip install tqdm
pip install h5py
pip install open3d
```

## 2. 数据预处理

### 2.1 下载3D-FRONT数据集
1. 访问[3D-FRONT官网](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)
2. 下载以下文件：
   - 3D-FRONT.json
   - 3D-FUTURE-model.json
   - 3D-FRONT-texture.zip

### 2.2 组织数据结构
```
Scene-Synthesis/
└── data/
    └── 3d-front/
        ├── 3D-FRONT.json
        ├── 3D-FUTURE-model.json
        └── texture/  # 解压3D-FRONT-texture.zip的内容
```

### 2.3 数据预处理步骤
1. 运行预处理脚本：
```bash
python scripts/preprocess_3dfront.py --data_dir data/3d-front
```

2. 预处理将生成以下文件：
```
data/3d-front/
├── processed/
│   ├── bedroom/
│   ├── living/
│   └── dining/
└── stats.json
```

## 3. 模型训练

### 3.1 配置训练参数
1. 创建配置文件：`configs/config.yaml`
```yaml
dataset:
  root: "data/3d-front/processed"
  room_types: ["bedroom"]  # 可选: bedroom, living, dining
  max_parts: 20
  batch_size: 32
  num_workers: 4

train:
  max_epochs: 100
  learning_rate: 0.001
  device: "cuda"
  checkpoint_dir: "checkpoints"
  log_interval: 100

model:
  latent_dim: 256
  hidden_dim: 512
```

### 3.2 开始训练
```bash
python train.py
```

训练日志和检查点将保存在：
- `checkpoints/`: 模型检查点
- `logs/`: 训练日志（如果使用wandb）

## 4. 模型测试

### 4.1 测试模型
```bash
python test.py --checkpoint checkpoints/model_best.ckpt
```

测试结果将保存在：
- `results/test/`: 重建结果可视化
- `results/test/metrics.txt`: 评估指标

### 4.2 生成新场景
```bash
python generate.py --checkpoint checkpoints/model_best.ckpt --num_samples 10
```

生成的场景将保存在：
- `results/generated/`: 生成的场景可视化

## 5. 结果说明

### 5.1 可视化结果
- `scene_xxxx.png`: 场景俯视图
- `scene_xxxx_3d.png`: 场景3D视图

### 5.2 评估指标
- 重建误差
- 覆盖率
- 多样性得分

## 6. 常见问题

1. 内存不足：
   - 减小batch_size
   - 减小max_parts数量

2. 训练不稳定：
   - 调整学习率
   - 检查数据预处理质量

3. 生成质量不佳：
   - 增加训练轮数
   - 调整模型架构参数

## 7. 引用
```
@inproceedings{3d-front,
    title={3D-FRONT: 3D Furnished Rooms with layOuts and semaNTics},
    author={Fu, Huan and Cai, Bowen and Gao, Lin and Zhang, Lingxiao and others},
    booktitle={ICCV},
    year={2021}
}
```
