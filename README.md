# SkinCancerSegmentation
Repository supporting the paper titled *"Interpretable Deep Learning Systems for Multi-Class Segmentation and Classification of Non-Melanoma Skin Cancer"* **(IN REVIEW)**

Authors:
- [Simon M. Thomas](https://orcid.org/0000-0003-4609-2732) **a, b**,
- [James G. Lefevre](https://orcid.org/0000-0002-5945-9575) **a**
- Glenn Baxter **b**
- [Nicholas A. Hamilton](https://orcid.org/000-0003-0331-3427) **a**

**a** - Institute for Molecular Bioscience, University of Queensland, 306 Carmody Road, St
Lucia, Australia

**b** - MyLab Pty. Ltd., 11 Hayling Street, Salisbury, Australia

<hr>

![Segmentation](./assets/whole_tissue_segmentation.png)

# Experimental Pipeline

The code presented here is intended for the experimental pipeline to be transparent. The data is a private collection and so is unpublished. The models
can be found in `seg_models.py`, and are subquently used in the appropriate enumerated script e.g. `06_model_evaluation.py`. Weights for the [2x](https://drive.google.com/drive/folders/1a3FSq65RHfDBJHhOXzVMEF5rVX0FDUpx?usp=sharing),
 [5x](https://drive.google.com/drive/folders/1t1AEYFdyklj1Xr3LnodLhols92WOECf4?usp=sharing) and [10x](https://drive.google.com/drive/folders/1ZfAD9R417BQkow_d76iwcaJLJ0doyLyg?usp=sharing) fine-tuned models are available, as are the CNN [classifier weights](https://drive.google.com/file/d/1VB6oqJfF5avvE8SSh86aLQIUt9xLnFhT/view?usp=sharing) 
(`10_classification.ipynb`). 


### Cancer Boundary Prediction

#### Basal Cell Carcinoma (IEC)

![BCC Demo](./assets/BCC.gif)

#### Intraepidermal Carcinoma (BCC)
![IEC Demo](./assets/IEC.gif)



