---
title: 模型量化方法简介
date: 2021-04-27 19:07:32
categories:
- 深度学习
tags:
- PyTorch
- 模型量化
coauthor: Zihe Wang
---

# 模型量化方法简介

模型量化可以理解为把模型中的 32 位浮点数参数压缩成 8 位整数，从而达到模型压缩的效果。目前的模型量化的方法仍然比较混乱，笔者亦了解不深。这里简单记录一下大致的方法，供后续优化时参考。

这里主要介绍模型量化的几种工具：

## PyTorch

PyTorch 提供了[官方的模型量化教程](https://pytorch.org/docs/stable/quantization.html)，但其内容混乱不堪。其模型量化的方法大致有以下三种：

1. 动态量化（Dynamic Quantization）：主要用来量化 NLP 的模型，如 LSTM（RNN 系列）、BERT（其实只量化其中的线性层）。
2. 静态量化（Static Quantization）：训练后的静态量化，主要用来对 CV 的模型进行量化压缩，主要量化其卷积层、ReLU 激活函数等。
3. 量化感知训练（Quantization Aware Training, QAT）：QAT 可能是 PyTorch 中准确率最高的量化方法。在使用 QAT 时，训练的正向和反向过程中，所有权重和激活函数都被伪量化。看起来是一个很好的量化方法，但似乎其支持的神经网络层很有限，文档亦混乱不清，可能是一个值得尝试的方法。

## Distiller

[Distiller](https://github.com/IntelLabs/distiller) 是由 Intel AI Lab 维护的基于 PyTorch 的开源神经网络压缩框架。主要包括：

- 用于集成剪枝（pruning），正则化（regularization）和量化（quantization ）算法的框架。
- 一套用于分析和评估压缩性能的工具。
- 现有技术压缩算法的示例实现。

## NNI

微软的 [NNI](https://github.com/microsoft/nni) 也提供了[模型压缩的模块](https://nni.readthedocs.io/zh/latest/model_compression.html)，其支持 Pytorch、Tensorflow、MXNet、Caffe2 等多个开源框架。

## TensorRT

[TensorRT](https://github.com/NVIDIA/TensorRT) 是 Nvidia 提出的神经网络推理（Inference）引擎，支持训练后 8bit 量化，它使用基于交叉熵的模型量化算法，通过最小化两个分布的差异程度来实现。尚未了解其是否有针对嵌入式设备的量化压缩方法。

## Tensorflow Lite

[TensorFlow Lite](https://www.tensorflow.org/lite/) 是谷歌推出的面向嵌入式设备的推理框架，支持 float16 和 int8 低精度，其中 8bit 量化算法细节可以参考白皮书 “Quantizing deep convolutional networks for efficient inference: A whitepaper”，支持训练后量化和量化感知训练，这也是大部分量化框架的算法原理。

同时谷歌也新推出了 [TensorFlow Model Optimization Toolkit](https://github.com/tensorflow/model-optimization)，目前包含了模型剪枝和量化两种 API。

## PaddleSlim

[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 是百度提出的模型量化工具，包含在 PaddlePaddle 框架中，支持量化感知训练，离线量化，权重全局量化和通道级别量化。

## 参考资料

1. [【杂谈】当前模型量化有哪些可用的开源工具？](https://zhuanlan.zhihu.com/p/98048208)

