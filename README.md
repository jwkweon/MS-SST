# MS-SST

Official Repo of "MS-SST: Single Image Reconstruction-based Stain-Style Transfer for Multi-domain Hematoxylin & Eosin Stained Pathology Images"

by
Juwon Kweon,
Mujung Kim,
Gilly Yun,
Soonchul Kwon,
and Jisang Yoo

> This paper has been submitted for publication in *IEEE Access*.


## Abstract

> .

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://github.com/jwkweon/MS-SST) repository:

    git clone https://github.com/jwkweon/MS-SST.git


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```


## Reproducing the results



## License



