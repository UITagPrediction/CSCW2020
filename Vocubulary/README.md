# UI Tag Semantics

This project aimed to distill the semantic association via [Apriori Algorithm](https://link.springer.com/content/pdf/10.1007/3-540-45372-5_2.pdf). 
In this work, we use the Louvain method implemented in the [Gephi](https://gephi.org/) tool to detect communities.

<div style="color:#0000FF" align="center">
<img src="https://github.com/UITagPrediction/CSCW2020/blob/master/figures/figure3.png"/> 
<figcaption>Fig. 1. The UI-related tag associative graph from December 27, 2018 to March 19, 2019.</figcaption>
</div>

## Installation

* Download the [Gephi](https://gephi.org/) for your platform

### Compilation

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

## Usage

```
python semantics.py
```

Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --support  	            | 0.003         |  minimum support threshold
| --confidence              |0.005          |  minimum confidence threshold
| --lift 		            |2           	|  minimum lift threshold
| --length                |   2        |  minimum length threshold
| -h --help                 |               | show this help message and exit


## License
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)