### Overview
This repository contains the source code for Group 10's ML in CompBio Final Project. The details of our project can be found in the abstract.

### Abstract 

An essential part of biological research and medical diagnostics is determining protein function. One main issue with current prediction methods is that they use a single modality, which limits their accuracy due to the narrow scope of information that is being used. To find a solution for this problem, we have created a multimodal machine learning framework that integrates sequence data from UniProt, 3D structures from Alphafold encoded by ProteinMPNN, and annotations from BioBERT. Each of these modalities was run through its specialized deep learning models. We used ProtT5 for the sequence data, neural networks for the structure embeddings, and transformers for the textual data. These predictions were then combined through a multi-headed attention neural network in order to predict the Enzyme Commission (EC) numbers. We used precision, recall, F1-score, and AUROC metrics in order to evaluate the performance of our models and to compare our ensemble model to each base model. We determined that our multi-modal approach showed a slight improvement in prediction accuracy, indicating that combining diverse protein data provides an advantage in protein function prediction.


### File Structure





### Report
* https://docs.google.com/document/d/1NPdxfBH7t91a6blf6zrmwZCzc-hwK66q9_cPZXo0UZA/edit?usp=sharing

