## Generalwise Separable Convolution
It is acknowledged that the depthwise separable convolution effectively reduces the computational complexity of
a standard convolution. However, its depthwise convolution only performs on the spatial domain while neglecting to consider other
domains such as the one formed by the channel and width/height dimensions. This paper specifically bridges the gaps by proposing
the generalwise separable convolution to generalize the depthwise separable convolution beyond the spatial domain to recruit the
widthwise and heightwise convolutions. A sequential combination of pointwise group convolution, channel shuffling, channel
splitting, and dimension transposing is required to implement the generalwise separable convolution. By embedding the generalwise
separable convolution into a stack of inverted residuals with linear bottlenecks, we propose GSCNet as a lightweight neural
backbone for various embedded vision tasks. Our empirical evidence indicates that the generalwise separable convolution
is superior to the depthwise separable convolution by feature extraction from domains complementing the spatial domain.
Experimental results show that GSCNet outperforms other stateof-the-art mobile CNNs over multiply vision tasks. On ImageNet
object classification benchmark, GSCNet achieves 75.5% top-1 accuracy with 216.98M multiply-adds, which is 28.1% fewer
than that of MobileNetv2 and 29.3% fewer than that of HBONet. GSCNet also yields better mAP quality than MobileNetv1/v2 and
MnasNet on COCO object detection benchmark.
<br>

<img width="1314" height="492" alt="image" src="https://github.com/user-attachments/assets/db9c4489-bf70-45a3-b26f-639bd35bb307" /> 


Given an input feature map $F \in ℝ^{H \times W \times C}$, we employ a pointwise $1 \times 1$ group convolution with group number of 3 to project the input to an expansion layer $E \in ℝ^{H \times W \times (TC)}$, where $T$ is the expansion ratio. A consecutive transformation of channel shuffling and equalized channel splitting loads equal subsets of the input tensor into three parallel branches: depthwise convolution, heightwise convolution, and widthwise convolution. For depthwise convolution, it works in the default setting. For heightwise and widthwise convolution, a dimension transposing operator swaps the height or width dimension with the channel dimension. Consequently, the height or width dimension becomes the “new” depth dimension with $3 \times 3$ convolutional filters executing feature extraction. Subsequently, another dimension transposing operation restores the output dimensions of the heightwise and widthwise convolution to match the output dimensions of the depthwise convolution. A concatenation operator gathers the outputs from all three of them. 
 Unlike depthwise separable convolution which receives all feature maps of a previous layer, each generalwise convolution only receive one-third of the input feature maps. Thus, we add residual connections from the tensor in the channel shuffling step to the concatenated tensor to resolve the challenge of not keeping all useful information about the input due to the zeroing out of ReLU activation. Finally, a $1 \times 1$ linear convolution without ReLU projects the concatenated tensor to an output feature map of $ℝ^{H \times W \times M}$, where $M$ is the output channel number. 

If this network structure inspires you, please cite us:
```
@article{Zu2022GeneralwiseSC,
  title={Generalwise Separable Convolution for Mobile Vision Applications},
  author={Shicheng Zu and Yucheng Jin and Yang Li},
  journal={2022 IEEE Symposium Series on Computational Intelligence (SSCI)},
  year={2022},
  pages={1074-1081}
}
```
