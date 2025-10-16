Crocodile Species Classification Project
This is my first machine learning project, built for learning purposes using a dataset from Kaggle. The goal is to classify crocodile species based on various features.
Project Overview
This project involves:

* Loading and preprocessing a crocodile species dataset from Kaggle.
* Feature selection and encoding.
* Training a K-Nearest Neighbors (KNN) classifier.
* Evaluating model performance.

 Libraries Used
Here are the libraries used in this project along with their versions (as of October 2025):
LibraryVersionPurposepandas2.0.3Data manipulation and analysisnumpy1.24.3Numerical computationsscikit-learn1.3.0Machine learning algorithmsmatplotlib3.7.1Data visualizationseaborn0.12.2Statistical data visualizationkagglehub0.1.0Downloading datasets from Kagglemglearn0.1.0Additional plotting utilities
To install these libraries, you can use pip:
  pip install pandas numpy scikit-learn matplotlib seaborn kagglehub mglearn
Getting Started
Prerequisites

* Python 3.8 or higher
* The libraries listed above

Installation

1. Clone the repository:

bashDownloadCopy codegit clone https://github.com/CyberPsychotic/crocodile-knn
cd crocodile-species-classification

1. Install the required libraries:

bashDownloadCopy codepip install -r requirements.txt
Running the Project
Simply run the Jupyter notebook or Python script containing the code:
bashDownloadCopy codepython crocodile_classification.py
 Dataset
The dataset used in this project is the Global Crocodile Species Dataset from Kaggle. It contains various features related to crocodile observations, such as:

* Age Class
* Sex
* Length
* Weight
* Habitat Type
* Country/Region
* Genus (target variable)

Challenges Faced:

1. Feature Selection: Deciding which features to keep and which to drop was challenging. Some features had too many unique values (e.g., Country/Region, Habitat Type), which could lead to overfitting or high dimensionality.

2. Choosing the Best Classification Model: Initially, I considered using a simple KNN model. However, I realized that other models like Random Forest or SVM might perform better. For simplicity, I stuck with KNN for this project but plan to explore other models in the future.

3. Data Preprocessing: Handling categorical variables (e.g., Sex, Age Class) required careful encoding to ensure the model could process them correctly.

4. Model Evaluation: Understanding how to evaluate the model's performance was crucial. I used metrics like accuracy, classification report, and confusion matrix to assess the model.


Results
The KNN model achieved an accuracy of approximately 86% on the test set. The confusion matrix and classification report provide more detailed insights into the model's performance.

Future Improvements

* Experiment with other classification models (e.g., Random Forest, SVM, Gradient Boosting).
* Address the Unknown values in Sex column.
* Address class imbalance if present using techniques like SMOTE.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Thank you for checking out my first machine learning project! I hope this helps others who are just starting their journey in ML.
