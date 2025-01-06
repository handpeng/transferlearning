import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones


class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class  # 分类的类别数
        self.base_network = backbones.get_backbone(base_net)  # 基础特征提取器
        self.use_bottleneck = use_bottleneck  # 是否使用瓶颈层来降低特征维度并对齐特征
        self.transfer_loss = transfer_loss  # 迁移损失函数
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        
        self.classifier_layer = nn.Linear(feature_dim, num_class)  # 用于将特征映射到类别空间，完成分类任务
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)  # 迁移学习的核心模块，实现源领域和目标领域之间的分布对齐 MMD、LMMD、DAAN
        self.criterion = torch.nn.CrossEntropyLoss()  # 分类损失函数，使用交叉熵损失（CrossEntropyLoss）

    def forward(self, source, target, source_label):

        # 对源领域和目标领域的数据分别通过基础网络提取特征
        source = self.base_network(source)
        target = self.base_network(target)

        # 如果使用瓶颈层，对提取的特征进一步变换
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)

        # classification
        source_clf = self.classifier_layer(source)
        clf_loss = self.criterion(source_clf, source_label)

        # transfer
        kwargs = {}

        # 一种基于标签的分布对齐方法
        if self.transfer_loss == "lmmd":
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)

        # 一种基于对抗训练的领域适配方法
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)

        # 一种迁移损失方法，目标是通过最小化目标领域特征的核范数（Nuclear Norm）来增强目标领域的特征紧凑性和判别性
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)
        
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass