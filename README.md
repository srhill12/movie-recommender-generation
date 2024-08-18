
# Movie Recommendation System

This repository contains a Jupyter Notebook and a Python script that together implement a collaborative filtering-based movie recommendation system. The system is designed to generate personalized movie recommendations for new users based on their previous ratings.

## Project Overview

The goal of this project is to develop a recommendation system that predicts movie ratings for users and recommends movies they might enjoy. The model is built using a Restricted Boltzmann Machine (RBM) implemented in TensorFlow. The project is divided into two main components:

1. **Jupyter Notebook**: This notebook demonstrates the process of generating recommendations for new users by leveraging pre-trained model weights and biases.
2. **Python Script (`utils.py`)**: This script contains utility functions that are used for data processing, model inference, and recommendation generation.

## Getting Started

### Prerequisites

To run the code, you need the following libraries installed:

- `pandas`
- `tensorflow`
- `numpy`

You can install these libraries using pip:

```bash
pip install pandas tensorflow numpy
```

### Dataset

The dataset used in this project consists of user ratings for movies. The data is loaded directly from CSV files hosted online. The two datasets used are:

1. **Original Ratings Data**: Contains user ratings for various movies.
2. **New Users Data**: Contains ratings from new users for a subset of movies.

### How to Run

1. **Jupyter Notebook**: 

   - Open the Jupyter Notebook file in your preferred environment (e.g., JupyterLab, VS Code, etc.).
   - Run the cells sequentially to generate movie recommendations for new users.

2. **Python Script (`utils.py`)**:

   - The Python script `utils.py` contains all the utility functions needed for data preprocessing, model weight retrieval, and recommendation generation.
   - This script is imported and used in the Jupyter Notebook.

### Key Functions

- **Data Preprocessing**:
  - `get_data()`: Loads and preprocesses the original ratings data.
  - `get_new_data()`: Loads and preprocesses the ratings data for new users.
  - `pivot_data()`: Pivots the ratings data into a user-item matrix format.
  - `normalize_data()`: Normalizes the ratings data to a scale of 0-1.

- **Model Utility Functions**:
  - `weights()`: Loads the pre-trained weights for the model.
  - `hidden_bias()`: Loads the pre-trained hidden layer biases.
  - `visible_bias()`: Loads the pre-trained visible layer biases.
  - `generate_recommendation()`: Generates movie recommendations for a given user based on their ratings.

### Example Usage

In the Jupyter Notebook, you can see an example of how to generate recommendations for a test user from the new users' dataset. The recommendations are generated using the following steps:

1. Load the new user ratings data.
2. Normalize the data.
3. Load the pre-trained model weights and biases.
4. Generate recommendations for the user.
5. Merge the recommendations with the original dataset to filter out movies that the user hasn't rated yet.

### Evaluation

The Root Mean Squared Error (RMSE) is calculated to evaluate the accuracy of the model's predictions. The RMSE measures the difference between the predicted and actual ratings, providing an indicator of the model's performance.

### Future Work

- Improve the model by fine-tuning hyperparameters.
- Incorporate additional features such as movie genres, user demographics, etc., to enhance recommendation accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this project is provided by an online course on AI and machine learning.
- TensorFlow and Pandas were instrumental in building and evaluating the model.

```
