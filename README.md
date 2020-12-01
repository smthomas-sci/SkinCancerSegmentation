<center>
 
 # [Medical Image Analysis] Interpretable Deep Learning Systems for Multi-Class Segmentation and Classification of Non-Melanoma Skin Cancer
 
 </center>

<center>

**Official repository of the paper**

</center>

Authors:
- [Simon M. Thomas](https://orcid.org/0000-0003-4609-2732) **a, b**,
- [James G. Lefevre](https://orcid.org/0000-0002-5945-9575) **a**
- Glenn Baxter **b**
- [Nicholas A. Hamilton](https://orcid.org/000-0003-0331-3427) **a**

**a** - Institute for Molecular Bioscience, University of Queensland, 306 Carmody Road, St
Lucia, Australia

**b** - MyLab Pty. Ltd., 11 Hayling Street, Salisbury, Australia

### Abstract
We apply for the first-time interpretable deep learning methods simultaneously to the most common skin cancers (basal cell carcinoma, squamous cell carcinoma and intraepidermal carcinoma) in a histological setting. As these three cancer types constitute more than 90% of diagnoses, we demonstrate that the majority of dermatopathology work is amenable to automatic machine analysis. A major feature of this work is characterising the tissue by classifying it into 12 meaningful dermatological classes, including hair follicles, sweat glands as well as identifying the well-defined stratified layers of the skin. These provide highly interpretable outputs as the network is trained to represent the problem domain in the same way a pathologist would. While this enables a high accuracy of whole image classification (93.6-97.9%), by characterising the full context of the tissue we can also work towards performing routine pathologist tasks, for instance, orientating sections and automatically assessing and measuring surgical margins. This work seeks to inform ways in which future computer aided diagnosis systems could be applied usefully in a clinical setting with human interpretable outcomes.

### Citation

```
@article{THOMAS2021101915,
title = "Interpretable deep learning systems for multi-class segmentation and classification of non-melanoma skin cancer",
journal = "Medical Image Analysis",
volume = "68",
pages = "101915",
year = "2021",
issn = "1361-8415",
doi = "https://doi.org/10.1016/j.media.2020.101915",
url = "http://www.sciencedirect.com/science/article/pii/S1361841520302796",
}
```

<hr>

![Segmentation](./assets/whole_tissue_segmentation.png)

# Experimental Pipeline

The code presented here is intended for the experimental pipeline to be transparent. The data is a private collection and so is unpublished. The models
can be found in `seg_models.py`, and are subquently used in the appropriate enumerated script e.g. `06_model_evaluation.py`. Weights for the [2x](https://drive.google.com/drive/folders/1a3FSq65RHfDBJHhOXzVMEF5rVX0FDUpx?usp=sharing),
 [5x](https://drive.google.com/drive/folders/1t1AEYFdyklj1Xr3LnodLhols92WOECf4?usp=sharing) and [10x](https://drive.google.com/drive/folders/1ZfAD9R417BQkow_d76iwcaJLJ0doyLyg?usp=sharing) fine-tuned models are available, as are the CNN [classifier weights](https://drive.google.com/file/d/1VB6oqJfF5avvE8SSh86aLQIUt9xLnFhT/view?usp=sharing) 
(`10_classification.ipynb`). 


### Cancer Boundary Prediction

#### Basal Cell Carcinoma (BCC)

![BCC Demo](./assets/BCC.gif)

#### Intraepidermal Carcinoma (IEC)
![IEC Demo](./assets/IEC.gif)



