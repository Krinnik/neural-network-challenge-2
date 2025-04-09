# Employee Attrition and Department Prediction Neural Network

## Overview

This file contains the code for a neural network designed to predict two outcomes for employees: their likelihood of leaving the company (attrition) as well as the department each employee might be best suited for. The model is built using TensorFlow and Keras.

## Preprocessing

The preprocessing involved:

1.  Separating 'attrition' and 'department' as target variables.
2.  Selecting at least 10 other columns as features.
3.  Splitting the data into training and testing sets.
4.  Converting feature data to numeric types (encoding categorical features).
5.  Scaling numerical features using `StandardScaler`.
6.  Encoding both 'department' and 'attrition' target variables using `OneHotEncoder`.

## Model Architecture

The network consists of:

1.  An **input layer** that accepts employee features.
2.  Two **shared hidden layers** with ReLU activation, intended to learn general patterns from the input data.
3.  A **branch for Department prediction**:
    * A hidden layer with 'tanh' activation.
    * An output layer with 'softmax' activation to predict the probability distribution across different departments.
4.  A **branch for Attrition prediction**:
    * A hidden layer with ReLU activation.
    * An output layer with 'sigmoid' activation to predict the likelihood of an employee leaving.

## Prediction Tasks

The model is trained to perform two simultaneous prediction tasks:

1.  **Attrition Prediction:** A binary classification problem (employee is likely to leave or not).
2.  **Department Prediction:** A multi-class classification problem (predicting the department that best fits each employee).

## Key Activation Functions

* **Attrition Output:** 'sigmoid' (for binary probability).
* **Department Output:** 'softmax' (for multi-class probability distribution).
* **Department Hidden Layer:** 'tanh' (user experimentation showed improvement over 'relu').
* **Shared Layers and Attrition Hidden Layer:** 'relu'.

## Evaluation

The model's performance was evaluated using accuracy for both tasks. The reported scores on the testing data were:

* Attrition predictions accuracy: ~0.87
* Department predictions accuracy: 0.6

## Summary Q&A
1. Is accuracy the best metric to use on this data? Why or why not?

* Accuracy might not be the best metric for both tasks, especially if there is a significant class imbalance in either the attrition predictions (far fewer employees leave than stay) or the department predictions (some departments might be much larger than others).

    * Attrition Predictions: While an accuracy of ~87% seems good, if the dataset is imbalanced, a model could achieve high accuracy by simply predicting the majority class most of the time. Metrics like precision, recall, F1-score, and the Area Under the ROC Curve (AUC) would provide a more comprehensive view of the model's performance, especially in identifying employees likely to leave.
    * Department Predictions: An accuracy of 0.6 (60%) might be acceptable depending on the number of departments. However, if some departments are much more frequent than others, the model might be biased towards predicting those. Again, metrics like precision, recall, and F1-score (potentially macro or weighted averages) for each department would give a better understanding of the model's ability to correctly classify employees into different departments. A confusion matrix would also be very helpful to see which departments are being misclassified.
#
2. What activation functions were chosen for output layers, and why?

    * For the Attrition predictions output layer, the activation function I used was 'sigmoid'. This is because attrition is a binary classification problem, and the sigmoid function outputs a probability between 0 and 1.
    * For the Department predictions output layer, the activation function I used is 'softmax'. This is because department prediction is a multi-class classification problem, and softmax provides a probability distribution over all possible departments.
#
3. Name a few ways that this model might be improved?

* Here are several ways the model could potentially be improved:


    * Hyperparameter Tuning: Experiment with different numbers of units in the shared and branch-specific hidden layers. Also, try different activation functions (e.g., 'relu' is common, but others like 'leaky_relu' or 'elu' could be explored).
    * Address Class Imbalance: If there's a significant imbalance in the attrition or department classes in your training data, consider techniques like oversampling the minority class, undersampling the majority class, or using class weights in the loss function.
    * Regularization: Add regularization techniques like L1 or L2 regularization to the dense layers or use dropout layers to prevent overfitting.
    * Increase Data: More data, if available, can often lead to better model performance.
    * Feature Engineering: Create new features from the existing data that might be more predictive of attrition or department.
    * Different Model Architectures: Explore slightly different network structures, perhaps adding more shared layers or adjusting the number of layers in the branches.
    * Learning Rate Tuning: Experiment with different learning rates for the optimizer (Adam).
    * Evaluate with More Metrics: As mentioned in question 1, use a wider range of metrics beyond just accuracy, such as precision, recall, F1-score, and AUC (for attrition), and per-class precision, recall, and F1-score, and a confusion matrix (for department).
