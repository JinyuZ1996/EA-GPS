# **EA-GPT** 

<p align="left">
  <img src='https://img.shields.io/badge/python-3.6+-blue'>
  <img src='https://img.shields.io/badge/tensorflow-1.12+-blue'>
  <img src='https://img.shields.io/badge/numPy-1.16+-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-0.20+-brightgreen'>
  <img src='https://img.shields.io/badge/scipy-1.5+-brightgreen'>
</p> 

## **Overall description** 
- Here presents the code of EA-GPT. The code is attached to our paper: " **Lightweight yet Efficient: An External Attentive Graph Convolution Network with Prompt-Tuning for Sequential Recommendation" (WWW 2024)** ". If you want to use our codes and datasets in your research, please cite our paper. If you want to follow our work, please consider to download the paper on Arxiv: [https://arxiv.org/abs/----.----](https://arxiv.org/abs/----.----). 
## **Code description** 
### **Vesion of implementations and packages**
1. python = 3.6.0
2. tensorflow-gpu = 1.12.0
3. tensorboard = 1.12.2
4. scipy = 1.5.2
5. pandas = 0.20.3
6. numpy = 1.16.0
### **Source code of EA-GPT**
1. The definition of main components see: EA_GPT/GPT_Model.py.
2. The parameter-settings see: EA_GPT/GPT_Config.py.
3. The training process see: EA_GPT/GPT_Train.py.
4. To run the training method see: EA_GPT/GPT_Main.py.
5. And the training log printer was defined in: EA_GPT/GPT_Printer.py.

-- The directory named Checkpoint is used to save the trained recommenders and  to evaluate the parameter scale.
