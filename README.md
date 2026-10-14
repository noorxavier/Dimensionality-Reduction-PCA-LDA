# Dimensionality Reduction and Data Reconstruction Using PCA and LDA From Scratch

## 📌 Project Overview

This project implements two dimensionality reduction techniques, **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)**, from scratch using Python and NumPy.

The objective is to reduce the dimensionality of a dataset from **D dimensions to K dimensions**, reconstruct the data into its original D-dimensional space, and calculate the reconstruction error.

Both techniques are implemented using fundamental mathematical operations, including covariance matrices, scatter matrices, eigenvalues, and eigenvectors, without using prebuilt PCA or LDA implementations from machine learning libraries.

## 🎯 Objectives

* Implement PCA from scratch using NumPy.
* Implement LDA from scratch using NumPy.
* Reduce the dimensionality of the dataset from D dimensions to K dimensions.
* Reconstruct the original data from the reduced representation.
* Calculate reconstruction error using Mean Squared Error (MSE) and Sum of Squared Errors (SSE).
* Visualize the reduced-dimensional data.
* Compare PCA and LDA based on their reconstruction errors.

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

## 📂 Dataset

The project uses a wine classification dataset containing numerical features and class labels.

The dataset used in this implementation contains:

* **Samples:** 178
* **Numerical features:** 13
* **Classes:** 3

The class labels are used by LDA to find discriminant directions. PCA does not use class labels when learning its principal components.

## 🧠 Methodology

### 1. Data Preprocessing

The dataset is loaded using Pandas, and the numerical features are separated from the class labels.

The features are standardized to have approximately zero mean and unit variance.

Standardization helps prevent features with larger numerical scales from dominating the dimensionality reduction process.

### 2. Principal Component Analysis (PCA)

PCA is an unsupervised dimensionality reduction technique that identifies directions of maximum variance in the dataset.

#### Implementation Steps

1. Center the dataset.
2. Calculate the covariance matrix.
3. Compute eigenvalues and eigenvectors.
4. Sort the eigenvalues in descending order.
5. Select the top K principal components.
6. Project the data into the reduced-dimensional space.
7. Reconstruct the original data.
8. Calculate reconstruction error.

#### Mathematical Formulation

The covariance matrix is calculated as:

$$
C = \frac{X_c^T X_c}{N-1}
$$

The reduced representation is:

$$
Z = X_c W
$$

The reconstructed data is:

$$
\hat{X} = ZW^T + \mu
$$

Here, \(W\) contains the selected principal components, and \(\mu\) is the mean vector removed during centering.

### 3. Linear Discriminant Analysis (LDA)

LDA is a supervised dimensionality reduction technique that identifies directions that help distinguish between different classes.

#### Implementation Steps

1. Calculate the overall mean.
2. Calculate the within-class scatter matrix.
3. Calculate the between-class scatter matrix.
4. Compute the eigenvalues and eigenvectors of the discriminant matrix.
5. Sort the eigenvalues in descending order.
6. Select the top K discriminant directions.
7. Project the data into the reduced-dimensional space.
8. Reconstruct the original data.
9. Calculate reconstruction error.

#### Mathematical Formulation

The within-class scatter matrix is:

$$
S_W = \sum_{c=1}^{C}\sum_{x_i \in c}
(x_i-\mu_c)(x_i-\mu_c)^T
$$

The between-class scatter matrix is:

$$
S_B = \sum_{c=1}^{C} N_c
(\mu_c-\mu)(\mu_c-\mu)^T
$$

The LDA directions are obtained by solving the generalized eigenvalue problem:

$$
S_Bw = \lambda S_Ww
$$

The reduced representation is:

$$
Z = (X-\mu)W
$$

Since the LDA projection matrix is not necessarily orthonormal, the reconstruction uses the pseudoinverse:

$$
\hat{X} = ZW^+ + \mu
$$

Here, \(W^+\) represents the Moore–Penrose pseudoinverse of the projection matrix.

For a dataset containing C classes, standard LDA can produce at most \(C-1\) discriminant directions.

Since the dataset contains three classes, the maximum number of LDA dimensions is two.

## 📊 Reconstruction Error

The reconstruction error measures the difference between the original data and the reconstructed data.

### Mean Squared Error (MSE)

$$
MSE = \frac{1}{ND}
\sum_{i=1}^{N}\sum_{j=1}^{D}
(X_{ij}-\hat{X}_{ij})^2
$$

### Sum of Squared Errors (SSE)

$$
SSE = \sum_{i=1}^{N}\sum_{j=1}^{D}
(X_{ij}-\hat{X}_{ij})^2
$$

A lower reconstruction error indicates that the reconstructed values are closer to the original values.

However, PCA and LDA optimize different objectives. PCA aims to preserve variance, whereas LDA aims to improve class separation.

Therefore, reconstruction error alone does not measure the overall classification usefulness of LDA.

## 📈 Visualization

The project generates visualizations to help analyze the results:

* PCA projection into two dimensions.
* LDA projection into two dimensions.
* Comparison of reconstruction errors.
* PCA reconstruction error for different numbers of retained components.

These visualizations help illustrate how dimensionality reduction affects the data representation.

## 📋 PCA vs LDA

| Feature              | PCA                          | LDA                                             |
| -------------------- | ---------------------------- | ----------------------------------------------- |
| Full form            | Principal Component Analysis | Linear Discriminant Analysis                    |
| Learning type        | Unsupervised                 | Supervised                                      |
| Uses class labels    | No                           | Yes                                             |
| Primary objective    | Maximize retained variance   | Improve class separation                        |
| Mathematical basis   | Covariance matrix            | Within-class and between-class scatter matrices |
| Maximum dimensions   | D                            | min(D, C - 1)                                   |
| Reconstruction error | Calculated using MSE and SSE | Calculated using MSE and SSE                    |

## 🚀 How to Run the Project

### Prerequisites

Install Python and the required libraries.

```bash
pip install numpy pandas matplotlib jupyter
```

### Steps

1. Clone the repository.

   ```bash
   git clone https://github.com/YOUR_USERNAME/Dimensionality-Reduction-PCA-LDA.git
   ```

2. Navigate to the project directory.

   ```bash
   cd Dimensionality-Reduction-PCA-LDA
   ```

3. Start Jupyter Notebook.

   ```bash
   jupyter notebook
   ```

4. Open the project notebook.

5. Ensure the dataset CSV is available in the correct location.

6. Run the notebook cells sequentially to execute preprocessing, PCA, LDA, reconstruction, and error analysis.

## 📁 Suggested Project Structure

```text
Dimensionality-Reduction-PCA-LDA/
│
├── data/
│   └── wine_dataset.csv
│
├── notebooks/
│   └── PCA_LDA_From_Scratch.ipynb
│
├── README.md
└── requirements.txt
```

The filenames above are suggested. Adjust them to match the actual files in your repository.

## 🔑 Key Learnings

* Understanding dimensionality reduction and feature transformation.
* Implementing PCA using covariance matrices and eigenvalue decomposition.
* Implementing LDA using within-class and between-class scatter matrices.
* Understanding the difference between supervised and unsupervised dimensionality reduction.
* Reconstructing data from a lower-dimensional representation.
* Evaluating reconstruction quality using MSE and SSE.
* Using NumPy for matrix operations and numerical linear algebra.

## 🔮 Future Improvements

* Compare reconstruction errors for different values of K.
* Evaluate how well PCA and LDA separate the different classes.
* Compare the from-scratch implementations with established machine learning library implementations.
* Experiment with additional datasets.
* Explore the relationship between retained dimensions, reconstruction error, and class separation.

## 👨‍💻 Author

**Your Name**

GitHub: [YOUR_GITHUB_USERNAME](https://github.com/YOUR_USERNAME)

---

*This project demonstrates the mathematical foundations and practical implementation of PCA and LDA for dimensionality reduction and data reconstruction.*
