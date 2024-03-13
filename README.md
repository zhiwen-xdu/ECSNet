
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ecsnet-spatio-temporal-feature-learning-for/event-data-classification-on-n-caltech-101)](https://paperswithcode.com/sota/event-data-classification-on-n-caltech-101)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ecsnet-spatio-temporal-feature-learning-for/event-data-classification-on-n-cars)](https://paperswithcode.com/sota/event-data-classification-on-n-cars)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ecsnet-spatio-temporal-feature-learning-for/event-data-classification-on-cifar10-dvs-1)](https://paperswithcode.com/sota/event-data-classification-on-cifar10-dvs-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ecsnet-spatio-temporal-feature-learning-for/gesture-generation-on-dvs128-gesture)](https://paperswithcode.com/sota/gesture-generation-on-dvs128-gesture)


<div align="center">
  <img src="assets/Logo.PNG" width="100%" higth="100%">
  <h3 align="center"><strong>ECSNet: Spatio-temporal feature learning for event camera [TCSVT '22] </strong></h3>
    <p align="center">
    <a>Zhiwen Chen</a><sup>1</sup>&nbsp;&nbsp;
    <a>Jinjian Wu</a><sup>1</sup>&nbsp;&nbsp;
    <a>Junhui Hou</a><sup>2</sup>&nbsp;&nbsp;
    <a>Leida Li</a><sup>1</sup>&nbsp;&nbsp;
    <a>Leida Li</a><sup>1</sup>&nbsp;&nbsp;
    <a>Guangming Shi<sup>1</sup></a>&nbsp;&nbsp;
    <br>
    <sup>1</sup>Xidian University&nbsp;&nbsp;&nbsp;
    <sup>2</sup>City University of Hong Kong&nbsp;&nbsp;&nbsp;

</div>



## About
The neuromorphic event cameras can efficiently sense the latent geometric structures and motion clues of a scene by generating asynchronous and sparse event signals. Due to the irregular layout of the event signals, how to leverage their plentiful spatio-temporal information for recognition tasks remains a significant challenge. Existing methods tend to treat events as dense image-like or point-serie representations. However, they either suffer from severe destruction on the sparsity of event data or fail to encode robust spatial cues. To fully exploit their inherent sparsity with reconciling the spatio-temporal information, we introduce a compact event representation, namely 2D-1T event cloud sequence (2D-1T ECS). We couple this representation with a novel light-weight spatiotemporal learning framework (ECSNet) that accommodates both object classification and action recognition tasks. The core of our framework is a hierarchical spatial relation module. Equipped with specially designed surface-event-based sampling unit and local event normalization unit to enhance the inter-event relation encoding, this module learns robust geometric features from the 2D event clouds. And we propose a motion attention module for efficiently capturing long-term temporal context evolving with the 1T cloud sequence. Empirically, the experiments show that our framework achieves par or even better state-of-the-art performance. Importantly, our approach cooperates well with the sparsity of event data without any sophisticated operations, hence leading to low computational costs and prominent inference speeds.
<div align="center">
  <img src="assets/Framework.PNG" width="80%" higth="80%">
</div>

## Getting Started
### Installation
Clone the repository locally:
```
pip install git+https://github.com/happychenpipi/ECSNet.git
```

Create and activate a conda environment and install the required packages:
```
conda create -n ecsnet python=3.7
conda activate ecsnet
bash install_ecsnet.sh
```

### Data Preparation
In this work, we evaluate our method on a wide range of event-based classification datasets, such as [N-MNIST](https://www.garrickorchard.com/datasets/n-mnist), [N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101), [N-Cars](https://www.prophesee.ai/2018/03/13/dataset-n-cars/), [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2) datasets and so on. Please download these data with the link below and put in ./data.


## Training
```
python ./train.py
```

## Evaluation
```
python ./test.py
```

## Acknowledgments
Thanks to [N-MNIST](https://www.garrickorchard.com/datasets/n-mnist), [N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101), [N-Cars](https://www.prophesee.ai/2018/03/13/dataset-n-cars/), [CIFAR10-DVS](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2) datasets, [PoimtMLP](https://github.com/ma-xu/pointmlp-pytorch) and [NVS2Graph](https://github.com/PIX2NVS/NVS2Graph) projects.

## Contact
Feedbacks and comments are welcome! Feel free to contact us via [zhiwen.chen@stu.xidian.edu.cn](zhiwen.chen@stu.xidian.edu.cn). 

## Citing ECSNet
If you use ECSNet in your research, please use the following BibTeX entry.

```
@article{chen2022ecsnet,
  title={Ecsnet: Spatio-temporal feature learning for event camera},
  author={Chen, Zhiwen and Wu, Jinjian and Hou, Junhui and Li, Leida and Dong, Weisheng and Shi, Guangming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={33},
  number={2},
  pages={701--712},
  year={2022},
  publisher={IEEE}
}
```
