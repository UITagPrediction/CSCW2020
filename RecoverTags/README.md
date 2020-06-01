# Recovering Tags in Dribbble design

## Sample Results

Figure 5 shows some predicted additional tags for example UI designs by our model. It shows that our models can help locate the platform (e.g., “website” for the second UI), screen color (e.g, “blue”for the fifth UI), app functionality (e.g., “food” for the first UI), screen functionality (e.g., “checkout”for the third UI), and screen layout (e.g., “grid” for the fourth example). All of these predicted tagsare not appearing in the original tag set, and these additional tags can complement with the originalones for more effective UI retrieval.

<div style="color:#0000FF" align="center">
    <img src="https://github.com/UITagPrediction/CSCW2020/blob/master/figures/figure9.png"/> 
<figcaption>Fig. 1. The predicted tags by our model for complementing the original tags.</figcaption>
</div>

<!-- ![alt text](/figures/figure1.png) -->

## Installation
* Pre-trained models: [models](https://drive.google.com/file/d/1UoX8DouGC8wZgGEKnH_TN97N5vF8j310/view?usp=sharing) 
* Glove model: [glove.6B.50d.txt](https://drive.google.com/file/d/1PMkFo6550XRHO7kocYx_wJFbelMKwTrV/view?usp=sharing)
* Resnet-18: [pre-trained resnet](https://drive.google.com/file/d/1t3UJ-kkf_6PkCImvdF-HBnGp0mPuwkYU/view?usp=sharing)

## Getting Started
* Put files in the following directory structure.

```
    .
    ├── train.py
    ├── demo.py
    ├── dataloader.py
    ├── categorization.py
    ├── model.py
    ├── glove.6B.50d.txt
    ├── requirements.txt
    ├── environment.yml
    └── Readme.md
```

### Compilation
**This work was tested with PyTorch 1.5.0, CUDA 10.2, python 3.5 and Ubuntu 16.04.**

* Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

* Install all the python dependencies using conda:
```
conda env create -f environment.yml
```

### Dataset setting
* Put your images in folder like this format
```

    .
    └── Image_Folder
       ├── Images
       |   ├── 1.jpg
       |   ├── .
       |   └── .
       └── Tags
           ├── 1.txt ("clean, creative, desert, interface, landing, ui, ...")
           ├── .
           └── .
```

### Training
```
python train.py --tag=sport --checkpoint_save_path=/tmp/real_weights --batch_size=32 --backbone=resnet --glove_path=/pretrain/glove.6B.50d.txt --train_path=/data/Dribbble/black --multigpu=0,1,2,3
```

* Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --tag  	                |	None        | tag to train
| --checkpoint_save_path  	|	None        | checkpoint state_dict file
| --resume                   |   100         | resume from checkpoint state_dict file
| --batch_size                   |   32         | batch size of training
| --lr  		            |   1e-4	    | initial learning rate
| --epoch                   |   100         | upper epoch limit
| --weight_decay  		            |   5e-3	    | initial weight decay
| --optim 	        |   Adam        | optimizer to use
| --backbone                   |   resnet         | backbone architecture
| --load_resnet_pretrain                   |   True         | Load resnet pretrain network
| --num_workers                   |   32         | Number of workers used in dataloading
| --glove_path                   |   None         | glove file
| --train_path                   |   None         | training data path
| --test_path                   |   None         | testing data path
| --feature_extract	            |   True          | feature exract on backbone
| --cuda                |   True           | Use CUDA to train model
| --multigpu                    |   1,2,3        | activate multi gpu
| -h --help                 |               | show this help message and exit

### Demo
* Put your images in folder like train dataset format
```
python demo.py --tag=sport --checkpoint_path=/tmp/real_weights/black.pth --demo_path=/data/Dribbble/black --result_path=/tmp/result --glove_path=/pretrain/glove.6B.50d.txt
```

## License
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)