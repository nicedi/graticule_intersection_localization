# Description

This is the source code for the paper:

> **Luan Dong**, Fengling Zheng, Hongxia Chang, Qin Yan. Corner points localization in electronic topographic maps with deep neural networks[J]Earth Science Informatics, 2018,11(1):47-57. <https://doi.org/10.1007/s12145-017-0317-3>

The provided files are the model(*locnet.py*), the training and testing program (*train.py* and *test.py* respectively). The format of ground-truth information can be figured out from the *.pkl* file.

# Dependencies

* Chainer
* Visdom (for visualizing the training process)
* OpenCV

# Usage

1. Scan your topographic maps. (We use a 200 dpi configuration)
2. Using **sloth** or any other labeling software to label the rectangle objects described in the paper.
3. Parse the .json file produced by sloth, and construct a ground-truth file likes the .pkl file in the codebase.