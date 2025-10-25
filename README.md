# Uncovering Nonlinear Patterns of Brain Insulin Resistance in Alzheimer's Disease Using Deep Representation Learning

**Author:** Iris Pang  
**Status:** In Progress  

## Overview
This project explores whether nonlinear transcriptomic patterns in the insulin signaling pathway can help characterize brain-specific insulin resistance in Alzheimer's disease. Using public RNA-seq data and autoencoder models, I aim to identify latent features that distinguish AD from control brain samples.

## Objectives
- Analyze expression of insulin signaling–related genes in Alzheimer’s brain tissue.  
- Train deep autoencoder models to learn representations of expression profiles.  
- Evaluate whether latent embeddings reflect disease status or known molecular dysregulation.
- Comparison with linear embeddings (i.e. PCA) to assess whether deep representations capture additional biological structure.

## Methods
- **Data:** AMP-AD/ROSMAP transcriptomic datasets (prefrontal cortex).  
- **Model:** Denoising autoencoder, PyTorch implementation.  
- **Analysis:** Latent feature visualization, correlation with clinical metadata.

## Status
- Literature review completed 
- Data preprocessing and model prototyping are underway
- Results and summary of context will be updated through a blog post

## Future Work
- Compare to alternative architectures (variational autoencoder, shallow network).  
- Interpret learned features relative to canonical insulin signaling pathways.

## Background and Related Work
Previous studies have suggested that insulin resistance in the brain contributes to Alzheimer's disease pathology through impairment of regular neuronal glucose uptake and signaling. Previous transcriptomic analyses have shown that dysregulation of insulin signaling molecules (IRS1, AKT1) is associated with cognitive function.
The application of omics approaches for studying brain insulin resistance is complicated due to the complexity of the disease. There has been emerging work applying deep learning to explore relationships between gene expression and disease state, as well as the application of neural networks to predict AD using transcriptomic data.
This project expands on these previous works to explore deeper insights between insulin resistance and AD.

**Selected References**
- de Oliveria Andrade, et al. (2024). *Brain insulin resistance and Alzheimer’s disease: a systematic review*
- Rhea, et al. (2024). *State of the Science on Brain Insulin Resistance and Cognitive Decline Due to Alzheimer’s Disease*
- Tong, et al. (2024). *Brain Insulin Signaling is Associated with Late-Life Cognitive Decline*
- Jackson, et al. (2013). *Clustering of transcriptional profiles identifies changes to insulin signaling as an early event in a mouse model of Alzheimer’s disease*

---
*This repository is part of an academic course project.*
