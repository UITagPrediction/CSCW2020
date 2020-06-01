```diff
- 1. Check any personal authorship
- 5. Vocubulary and Recovertags Readme.md add html image
```
# From Lost to Found: Discover Missing UI Designs through Recovering Missing Semantics

Design sharing sites provide UI designers with a platform to share their works and also an opportunity to get inspiration from others' designs. To facilitate management and search of millions of UI design images, many design sharing sites adopt collaborative tagging systems by distributing the work of categorization to the community.

However, designers often do not know how to properly tag one design image with compact textual description, resulting in **unclear, incomplete, and inconsistent** tags for uploaded examples which impede retrieval, according to our empirical study and interview with four professional designers. 

Based on the deep neural network, we introduce **a novel approach for encoding both the visual and textual information** to recover the missing tags for existing UI examples so that they can be more easily found by text queries. We achieve 82.72\% accuracy in the tag prediction. Through a simulation test of 5 queries, our system on average returns hundreds of times more results than the default Dribbble search, leading to better relatedness, diversity and satisfaction.	

## Details
To demonstrate our task, we first show some examples to illustrate what is the fundamental limitations in existing design sharing platforms.

Figure 1 shows an example design from the design sharing website Dribbble of which tags illustrate the two problems with tagging-based search. 

<div style="color:#0000FF" align="center">
<img src="figures/figure1.png"/> 
<figcaption>Fig. 1. An example design from the design sharing website http://www.dribbble.com/ of which tags illustrate the two problems with tagging-based search.</figcaption>
</div>

First, users may use different words to describe the same UI design based on their own background knowledge. The increase in the use of incoherent tags may hinder the content searching or navigation.
For example, when designers want to search for UI designs with "user interface", this design tagged with "ui" will be omitted.


Second, users may extract different topics from the same design according to their own understanding, hence missing some closely related tags associated with the GUI design images. 
When tagging the UI design, users tend to provide only a few keywords that describe the most obvious visual or semantic contents, while content in other aspects may be simply omitted.
For example, although this design is related to the "food" category, this label is missing from the tags.

Some example UIs with complete or incomplete tags from these designers can be seen in Figure 2.
<div style="color:#0000FF" align="center">
<img src="figures/completeandincomplete.png"/> 
<figcaption>Fig. 2. Examples of UI examples with complete&incomplete tags.</figcaption>
</div>
<!-- ![UI-related tags association graph](/figures/communitydetection.png) -->

## EMPIRICAL STUDY OF DESIGN SHARING
In this work, we select Dribbble as our study subject. Since its foundation in 2009, Dribbble has become one of the **largest** networking platforms for designers to share their own designs and get inspirations from others'.
Only professional designers can be invited to upload their design works for sharing with the community, leading to the **high-quality** design within the site.

### DATASET
We build a [web crawler](Crawl/README.md) to collect designs and associated metadata from Dribbble from December 27, 2018 to March 19, 2019 with a collection of **226,000** graphical design.

We show some examples in our dataset, top is the designs and bottom is its metadata.

<!-- <div style="color:#0000FF" align="center"> -->
<p align="center">
<img src="figures/figure2.png" width="70%"/> 
</p><p align="center">Fig. 3. Example and Metadata of our Dribbble dataset crawled from December 27, 2018 to March 19, 2019<p align="center">
<!-- </div> -->


<!-- 
Within the Dribbble site, the design creators can add at most 20 tags for their design work.  -->
<!-- ```diff
+ add statistic for dataset and show UI importance. Among the top 30 most common tags, approximately 25% are UI related (e.g., “ui”, “ux”, “app”, “web”, “interface”, etc.), which indicates that user interface design is one of the most popular design areas on Dribbble.
``` -->

The full dataset can be downloaded via [Dataset](https://drive.google.com/open?id=1UpoAxyY66zlRlJ7z4ZfZUWu_FDpPRhRb) ||
[Metadata](https://drive.google.com/file/d/1-xci75k3yZWxbb1BjK-kEg_HYH5VdDDU/view?usp=sharing)


### Overview of UI semantics

We adopt the [Association Rule Mining and Community Detection](Vocubulary/README.md) for visualizing the landscape of UI tags. Figure 4 shows the UI-related tag associative graph.
<div style="color:#0000FF" align="center">
<img src="figures/figure3.png"/> 
<figcaption>Fig. 4. The UI-related tag associative graph from December 27, 2018 to March 19, 2019</figcaption>
</div>

### Vocabulary of UI semantics
We adopted a consensus-driven, iterative approach to combine the observed tag landscape with existing expert knowledge documented inbooks and websites such as Mobile Design Pattern Gallery and Google’s Material Design.

Figure 5 shows the categorization of some most frequent UI-related tags. For example, the APP FUNCTIONALITY category contains "MUSIC", "FOOD & DRINK", "GAME", and the subcategory "FOOD & DRINK" contains UI design tagged with "Food", "Restaurant", "Drink", etc.
<p align="center">
<img src="figures/figure4.png" style="width:100%"/> 
<figcaption></figcaption>
</p><p align="center">Fig. 5. The categorization of some most frequent UI-related tags.<p align="center">

### Consistency of Vocabulary
We adopt a semi-automatic method (1) train a word embedding to extract semantically-related words like "minimal" and "minimalistic" (2) define a set of rules to discriminate the abbreviations (3) manually check the vocabulary

Figure 6 shows the 40 most frequent UI related tags with their abbreviations and synonyms and in brackets indicate the number of occurence.
<p align="center">
<img src="figures/figure5.png"/> 
</p><p align="center">Fig. 6. The 40 most frequent UI related tags with their abbreviations and synonyms and in brackets indicate the number of occurence.<p align="center">

The full UI category can be viewed [Here](RecoverTags/categorization.py)

## AUGMENT TAGS FOR THE UI DESIGN

Figure 7 shows the overview of our approach.
We first collect all existing UI design with specific tags identified in our empirical study, and then develop a binary tag prediction model (predicting the image is or isn't belonging to the tag) by combining a CNN model for capturing visual UI information and a fully-connected neural network for capturing textual information of existing tags.
Additionally, to understand how our ensemble model make its decisions through the visual information, we apply a visualization technique (Saliency Maps) for understanding which part of the figure and which keyword leading to the final prediction.
<p align="center">
<img src="figures/CNN_structure.png"/> 
<figcaption>Fig. 7. The architecture of our tag prediction model.</figcaption>
</p>

### Dataset preparing
Figure 8 shows the statistics of our dataset for each tag. The dataset contains 50% positive and 50% negative samples. 

<p align="center">
<img src="figures/dataset.png"/> 
<figcaption>Fig. 8. The number of instances per tag in the proposed dataset.</figcaption>
</p>

### Training and Demo process
Please follow the [Readme.md](RecoverTags/README.md) instruction in RecoverTags folder.

## EVALUATION
Note that as the splitting ratio may influence the final results, we experiment four splitting ratio (training : validation : tesing), 50%:25%:25%, 60%:20%:20%, 70%:15%:15% and 80%:10%:10% for each model respectively.

We further set up several basic machine-learning baselines including the feature extraction (e.g., color histogram) with machine-learning classifiers (e.g., decision tree, SVM). We further set up different settings of data splitting. 

**Results show that the improvement of our model is significant in all comparisons and in all data splitting.**

<p align="center">
<img src="figures/result.png"/> 
<figcaption>Fig. 9. Tag classification accuracy for four dataset splitting ratio in different methods.</figcaption>
</p>

The detailed results can be viewed here.
<p align="center">
<img src="figures/detailresult.png"/> 
<figcaption>Fig. 10. Tag classification accuracy in four splitting ratio.</figcaption>
</p>

Figure 11 shows some predicted additional tags for example UI designs by our model.
<p align="center">
<img src="figures/figure9.png"/> 
<figcaption>Fig. 11. The predicted tags by our model for complementing the original tags.</figcaption>
</p>

Figure 12 shows the visualization of salient visual and textual features in our model leading to the final predictions.
<p align="center">
<img src="figures/figure10.png"/> 
<figcaption>Fig. 12. Visualization of the salient features in our model leading to the final predictions.</figcaption>
</p>

 

Some common causes for tag augmentation failure. 
<p align="center">
<img src="figures/failure.png"/> 
<figcaption>Fig. 13. Examples of the three kinds of prediction errors.</figcaption>
</p>

## RETRIEVAL EVALUATION
We conduct a pilot user study to evaluate the usefulness of the predicted addition tags for boot-strapping the UI design retrieval. Figure 14 and Figure 15 provides initial evidence of the usefulness of our method for enhancing the performance of tagging-based search. For more detail of this user study, please [see the website](https://sites.google.com/view/uitagpredictionuserstudy/home).
<p align="center">
<img src="figures/table3.png"/> 
<figcaption>Fig. 14. The random queries for searching UI designs.</figcaption>
</p>

<p align="center">
<img src="figures/table4.png"/> 
<figcaption>Fig. 15. The comparison of the experiment and control groups. ∗ denotes 𝑝<0.01, ∗∗ denotes 𝑝<0.05.</figcaption>
</p>

## License
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)