### Overview
This repository contains the source code for Group 10's ML in CompBio Final Project. The details of our project can be found in the abstract.

### Abstract 

An essential part of biological research and medical diagnostics is determining protein function. One main issue with current prediction methods is that they use a single modality, which limits their accuracy due to the narrow scope of information that is being used. To find a solution for this problem, we have created a multimodal machine learning framework that integrates sequence data from UniProt, 3D structures from Alphafold encoded by ProteinMPNN, and annotations from BioBERT. Each of these modalities was run through its specialized deep learning models. We used ProtT5 for the sequence data, neural networks for the structure embeddings, and transformers for the textual data. These predictions were then combined through a multi-headed attention neural network in order to predict the Enzyme Commission (EC) numbers. We used precision, recall, F1-score, and AUROC metrics in order to evaluate the performance of our models and to compare our ensemble model to each base model. We determined that our multi-modal approach showed a slight improvement in prediction accuracy, indicating that combining diverse protein data provides an advantage in protein function prediction.


### File Structure
- `Annotation_Classification.ipynb`: Code to run the RNN annotation classifier based on the BioBERT annotation embeddings
- `Annotation_Data_Processing.ipynb`: Code to run the processing of the annotation data into embeddings via BioBERT (Uses code from [this repository](https://github.com/Overfitter/biobert_embedding) to extract sentence embeddings from the annotations
- `Ensemble Classifier.ipynb`: Code to run the multi-head attention ensemble classifier using the output vectors of class label weights as input
- `Parsed_EC_Descriptor.tsv`: Tsv file that includes information about the proteins we used in the analysis (includes UniProtID)
- `Sequence_Classifier.ipynb`: Code to run parameter-efficient fine-tuning (LoRA of ProtT5 model) for sequence-based classification (Adapted from [this paper](https://www.nature.com/articles/s41467-024-51844-2), source notebook can be found [here](https://zenodo.org/records/12770310))
- `Sequence_Data_Processing.ipynb`: Code to run data processing for the sequence data (fastas to CSV of sequences and labels)
- `Structure_Classifier.ipynb`: Code to run the transformer structure classifier based on the ProteinMPNN embeddings
- `finetune.yml`: Environment to run the finetuning notebook(`Sequence_Classifier.ipynb`)
- `protein_mpnn_run.py`: Code to process PDBs into ProteinMPNN embeddings (Adapted from [this paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9997061/) source python script can be found [here](https://github.com/dauparas/ProteinMPNN))



