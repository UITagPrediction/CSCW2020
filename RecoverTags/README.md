# Recovering Tags in Dribbble design

<!-- ## Overview
Graphical User Interface (GUI) is ubiquitous, but good GUI design is challenging and time-consuming. Despite the enormous amount of UI designs existed online, it is still difficult for designers to efficiently find what they want, due to the gap between the UI design image and textural query. To overcome that problem, design sharing sites like Dribbble ask users to attach tags when uploading tags for searching. However, designers may use different keywords to express the same meaning or miss some keywords for their UI design, resulting in the difficulty of retrieval. This project introduces an automatic approach to recover the missing tags for the UI, hence finding the missing UIs. Through an iterative open coding of thousands of existing tags, we construct a vocabulary of UI semantics with high-level categories. -->

## Getting Started
Put files in the following directory structure.

    .
    ├── Data  
    |   ├── All_images 
    |   ├── Metadata.csv
    |   ├── glove.6B.50d.txt (Not included)
    |   ├── generate_dataset.py
    |   ├── autoaugment.py
    |   └── categorization.py
    ├── Src
    ├── Baselines
    |   ├── CNN
    |   ├── ResNet
    |   └── CV
    ├── Demo
    ├── requirements.txt
    └── Readme.md

## Installation

* Download the [glove.6B.50d.txt](https://drive.google.com/open?id=1ublNdoeX8i5iTmwP_F-C1jS3SOzcHFT8)

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

## Usage
1. Generate dataset on tag (default: blue)
```
cd Data
python3 generate_dataset.py
```
2. Train the network on tag (default: blue)
```
cd Src
python3 train.py
```

Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --tag  	                |	blue        | model/tag to train
| --epoch                   |   100         | upper epoch limit
| --lr  		            |   1e-4	    | initial learning rate
| --weight_decay  		            |   5e-3	    | initial weight decay
| --optim 	        |   Adam        | optimizer to use
| --seed	            |   42          | random seed
| --model                |   resnet           | model architecture on visual information
| --mode                    |   true        | finetune the partial model
| --pretrain                |   true        | use pretrain network
| -h --help                 |               | show this help message and exit

3. Run the demo on tag to see the results 
```
cd Demo
```
* put your test image in **"./images/1/"** and metadata as **"Meta.csv"**
```
python3 test.py
```

## Authorship

This project is contributed by [Sidong Feng](https://github.com/u6063820).

## License
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)