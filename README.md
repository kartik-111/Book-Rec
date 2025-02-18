
# Recommendation System Using Multiple Techniques

## Overview
This project implements various **Recommendation System** techniques to predict user preferences based on past interactions. The system incorporates **Content-Based Filtering, Collaborative Filtering, Hybrid Filtering, and Matrix Factorization** methods to improve recommendation accuracy. Additionally, a **Feedforward Neural Network** is implemented for deep learning-based recommendations.

## Features
- **Content-Based Filtering**: Recommends items similar to those a user has previously interacted with using **TF-IDF vectorization** and **Cosine Similarity**.
- **Collaborative Filtering**: Uses user-item interaction data to recommend products using **SVD++**.
- **Hybrid Filtering**: Combines content-based and collaborative filtering for enhanced personalization.
- **Matrix Factorization**: Implements **SVD-based approaches** (with and without bias) to capture latent user-item relationships.
- **Feedforward Neural Network**: Uses **PyTorch** to model complex user-item interactions with deep learning.

## Techniques Used
1. **Content-Based Filtering**
2. **Collaborative Filtering**
3. **Hybrid Filtering (Content-Based + Collaborative Filtering)**
4. **Matrix Factorization without Bias (SVD, ALS)**
5. **Matrix Factorization with Bias (SVD++)**
6. **Feedforward Neural Network**

## Datasets Used
- `train.csv`: Contains user-item interactions and ratings.
- `books_metadata.csv`: Metadata including item details such as reviews and ratings.

## Evaluation Metrics
The models are evaluated using **RMSE (Root Mean Square Error), MAE (Mean Absolute Error), and MSE (Mean Squared Error)**.

| **Technique**  | **Evaluation Metric** | **Expected Performance** |
|---------------|----------------------|--------------------------|
| **Content-Based Filtering** | No RMSE/MAE | Good for niche users, suffers from cold-start issues |
| **Collaborative Filtering (SVD++)** | RMSE & MAE | **RMSE ~ 0.85 - 1.2**, better for dense data |
| **Hybrid Filtering** | Indirect evaluation | Improves diversity and personalization |
| **Matrix Factorization (No Bias)** | RMSE & MAE | **Higher RMSE (~1.0 - 1.3)** |
| **Matrix Factorization (With Bias)** | RMSE & MAE | **Lower RMSE (~0.85 - 1.1)** |
| **Feedforward Neural Network** | MSE Loss | Loss **~0.8 - 1.2**, requires more data |

## Dependencies
Install required dependencies before running the project:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk scipy torch
