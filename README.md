# Code for the paper "A Step Towards Inherently Interpretable Causal Machine Learning Models for Decision-Making"

This repository contains code, visuals, and experiments related to the manuscript.

## Environment Setup

To ensure reproducibility:
- Create a virtual environment with Python version 3.11.13
- Install dependencies:
  - pip install -r requirements.txt

## Repository Structure

- 001_general_visuals contains visuals from the paper that are not part of any experiments.
- 002_demonstration includes the experimental setup for the production use-case demonstration with results.
- 003_benchmark provides benchmarking experiments using external data sourced from the "https://github.com/microsoft/csuite" repository for further validation. Here, the results for the individual experiments are shown. The datasets are located in data/silver. Experiments are: nonlin_simpson, symprod_simpson, large_backdoor, and weak_arrows.
- The utils.py file contains the training and testing approaches for CML, preprocessing functions, and evaluation metrics.
- The hyperparameter.py file contains the models and the hyperparameters.
- The simulation_production.py file contains the simulation to generate data for the demonstration.
