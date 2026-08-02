# Streamlit Visualisation Tool for Logistic Regression
# I modified/wrote this script to dynamically see how hyperparameters (C, solver, penalty) affect the decision boundary!

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Helper function to generate data and plot the initial scatter of points
def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        # Generate 2 clusters for binary classification
        X_data, y_data = make_blobs(n_features=2, centers=2, random_state=6)
        ax.scatter(X_data.T[0], X_data.T[1], c=y_data, cmap='rainbow')
        return X_data, y_data
    elif dataset == "Multiclass":
        # Generate 3 clusters for multiclass classification
        X_data, y_data = make_blobs(n_features=2, centers=3, random_state=2)
        ax.scatter(X_data.T[0], X_data.T[1], c=y_data, cmap='rainbow')
        return X_data, y_data
    else:
        raise ValueError("Invalid dataset type specified")

# Helper function to generate a grid of points so we can draw the decision boundary
def draw_meshgrid(X):
    # Create ranges for X and Y coordinate axes, slightly padded so we cover all points
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    # Flatten the grids and transpose to get list of coordinate pairs
    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array



# Choose a nice style for our matplotlib plots
plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Logistic Regression Classifier")

# --- Sidebar Inputs for the user to tune hyperparameters ---

dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Binary', 'Multiclass')
)

penalty = st.sidebar.selectbox(
    'Regularization',
    ('l2', 'l1', 'elasticnet', 'none')
)

# C is the inverse of regularization strength - smaller values specify stronger regularization
c_input = float(st.sidebar.number_input('C', value=1.0))

solver = st.sidebar.selectbox(
    'Solver',
    ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
)

max_iter = int(st.sidebar.number_input('Max Iterations', value=100))

multi_class = st.sidebar.selectbox(
    'Multi Class',
    ('auto', 'ovr', 'multinomial')
)

l1_ratio = float(st.sidebar.number_input('l1 Ratio', min_value=0.0, max_value=1.0, value=0.0, step=0.1))

# Initialize plot figure
fig, ax = plt.subplots()

# Load initial graph based on dataset selection and split to train/test
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

# Let's train the model when the user clicks 'Run Algorithm'
if st.sidebar.button('Run Algorithm'):
    orig.empty() # Clear the old plot

    # Map 'none' string selection to Python's None object for newer scikit-learn versions
    actual_penalty = None if penalty == 'none' else penalty
    
    # l1_ratio is only used/valid for elasticnet in scikit-learn. Otherwise, keep it None to avoid warnings/errors.
    actual_l1_ratio = l1_ratio if penalty == 'elasticnet' else None

    try:
        # Initialize sklearn's LogisticRegression classifier with our inputs
        clf = LogisticRegression(
            penalty=actual_penalty, 
            C=c_input, 
            solver=solver, 
            max_iter=max_iter, 
            multi_class=multi_class, 
            l1_ratio=actual_l1_ratio
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Get grid points and predict their labels to draw decision regions
        XX, YY, input_array = draw_meshgrid(X)
        labels = clf.predict(input_array)

        # Plot the decision boundaries using contourf
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        plt.xlabel("Col1")
        plt.ylabel("Col2")
        orig = st.pyplot(fig)
        
        # Note: Fixing copy-paste label typo. It used to say "Accuracy for Decision Tree" but we're doing Logistic Regression!
        st.subheader("Accuracy for Logistic Regression: " + str(round(accuracy_score(y_test, y_pred), 2)))
    
    except Exception as e:
        st.error(f"Oops! The selected solver and penalty combination is incompatible. Error details: {e}")


