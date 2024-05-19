import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy import mean
from numpy import std
import streamlit as st
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
import io
from imblearn.over_sampling import RandomOverSampler
import plotly.graph_objects as go
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
st.set_option('deprecation.showPyplotGlobalUse', False)

logo_url = "https://logowik.com/content/uploads/images/ytu-yildiz-technical-university1208.logowik.com.webp"

# Markdown string with university logo
markdown_string = f"<img src='{logo_url}' alt='University Logo' width='250' style='float:left;margin-right:10px;'> <h1 style='font-size: 24px;'>Web Data Mining School Project</h1>"

# Display Markdown with logo
st.markdown(markdown_string, unsafe_allow_html=True)
#st.markdown("<h1 style='font-size: 24px;'>YTU Web Data Minig School Project</h1>", unsafe_allow_html=True)
project_url = "https://raw.githubusercontent.com/smt07/webdatamining/main/data-mining-project.webp"
st.image(project_url, caption="Samet Gök       Student ID: 23550030", use_column_width=True)
st.markdown("<h1 style='font-size: 24px;'>Online Shoppers Purchasing Intention Data Set</h1>", unsafe_allow_html=True)
dataset_name = "Online Shoppers Intention"

# Select Classifier 
st.sidebar.markdown(
    '<div style="font-weight: bold; font-style: italic; color: darkblue">Select Classifier</div>',
    unsafe_allow_html=True
)

classifier_name = st.sidebar.selectbox(
    "",
    ("Decision Tree Classifier", "Random Forest Classifier", "Naive Bayes Classifier")
)

url_name = 'https://github.com/smt07/webdatamining/blob/main/online_shoppers_intention.csv?raw=true'
data = pd.read_csv(url_name)

dataset_name = "Online Shoppers Intention Target Column: Revenue "

st.write("""
The dataset consists of feature vectors belonging to 12,330 sessions. 
The dataset was formed so that each session would belong to a different user in a 
1-year period to avoid any tendency to a specific campaign,
special day, user profile, or period. Of the 12,330 sessions in the dataset, 
84.5% (10,422) were negative class samples that did not end with shopping, 
and the rest (1908) were positive class samples ending with shopping.""")
st.write(data.head())

# Changed Revenue to dtype str
data['Revenue'] = data['Revenue'].astype('str')

# Replace True and False in Column Revenue by Sale or No Sale 
data['Revenue'] = data['Revenue'].replace(['True'],'Sale')
data['Revenue'] = data['Revenue'].replace(['False'],'No Sale')

# Write dataset name to Streamlit
Sub_Title = dataset_name
st.write(Sub_Title)

# Set the style
sns.set(style="whitegrid")

# Create the countplot
plt.figure(figsize=(6, 4)) 
ax = sns.countplot(x="Revenue", data=data, palette="Set2")

total = len(data["Revenue"])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 0.5
    ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=10, color='black')


plt.xlabel("Revenue", fontsize=12) 
plt.ylabel("Count", fontsize=12)  
plt.title("Distribution of Revenue", fontsize=14) 

# Customize background
ax.set_facecolor('#f7f7f7') 
ax.grid(True, color='white') 

sns.despine()

# Show the plot
plt.tight_layout()
plt.show()

st.pyplot()
# Discription about dataset
def get_dataframe_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return s

def create_info_table(df):
    info_dict = {
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Dtype": df.dtypes
    }
    info_df = pd.DataFrame(info_dict)
    return info_df
st.subheader("Dataset Information")
info_table = create_info_table(data)
st.dataframe(info_table)

# Split data into train and test
X = data.drop("Revenue", axis=1)
X = pd.get_dummies(X)

# Create our target variable
target=["Revenue"]
y = data.loc[:, target].copy()

# st.write("X: ", X)
# st.write("y: ", y)

# Make the train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# X, y = get_dataset(dataset_name)
st.write("Shape of Dataset Rows and Columns ", data.shape, "Number of Classes, Sale or No Sale  ", len(np.unique(y)))

# Model Selection
if classifier_name == "Decision Tree Classifier":
    st.subheader("Model: Decision Tree Classifier")
    image_url = "https://raw.githubusercontent.com/smt07/webdatamining/main/desicion%20tree%20classifier%20picture.png"
    st.image(image_url, caption="Decision Tree Classifier", use_column_width=True)

    st.write("""The decision tree classifier; creates the classification model 
    by building a decision tree. Each node in the tree specifies a test on an attribute, each branch descending from 
    that node corresponds to one of the possible values for that attribute. """)
    Random_State_Seed = 1

    # Resample the training data with the RandomOversampler
    strategy = 'auto'  
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=Random_State_Seed)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    st.write ("""Resampling involves creating a new transformed version of the training 
    dataset in which the selected examples have a different class distribution.
    This is a simple and effective strategy for imbalanced classification problems.""")
    st.write("Shape of resampled dataset with onehot encoding and Random Over Sapmling",
    X_resampled.shape, "Number of classes", len(np.unique(y_resampled)))

    # Select the hyperparameters for the Decision Tree
    max_depth_explanation = """
    The maximum depth of the tree represents the maximum number of splits it can make before reaching a decision. 
    where a higher depth allows for more intricate patterns to be captured 
    but increases the risk of overfitting to the training data.
    """

    st.sidebar.write(max_depth_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select max depth of the Decision Tree</div>',
        unsafe_allow_html=True
    )

    # Select the max depth of the Decision Tree
    max_depth = st.sidebar.select_slider(
        '',
        options=range(1, 21),  # Assuming the depth ranges from 1 to 20
        value=5  # Default value
    )

    criterion_explanation = """
    The criterion determines the function to measure the quality of a split in the decision tree.
    - Gini: Measures the impurity of the nodes. It tends to create smaller trees.
    - Entropy: Measures the information gain based on the reduction of entropy after the split.
    - Log Loss: Uses logistic regression's loss function.
    """

    st.sidebar.write(criterion_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select criterion for the Decision Tree</div>',
        unsafe_allow_html=True
    )

    # Select the criterion for the Decision Tree
    criterion = st.sidebar.selectbox(
        '',
        options=["gini", "entropy", "log_loss"],  
        index=0  # Default is "gini"
    )

    splitter_explanation = """
    The splitter strategy determines how the decision tree chooses the best split at each node.
    - Best: Selects the best split based on the chosen criterion.
    - Random: Randomly selects the best split from a random subset of features.
    """

    # Add an explanation section
    st.sidebar.write(splitter_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select splitter strategy</div>',
        unsafe_allow_html=True
    )

    # Select the splitter strategy for the Decision Tree
    splitter = st.sidebar.selectbox(
        '',
        options=["best", "random"], 
        index=0  
    )

    min_samples_split_explanation = """
    The min_samples_split parameter determines the minimum number of samples required to split an internal node. 
    - A smaller value allows the tree to consider more splits, potentially creating a deeper tree.
    - A larger value can prevent overfitting by limiting the tree's growth.
    """
    st.sidebar.write(min_samples_split_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select min samples split</div>',
        unsafe_allow_html=True
    )

    # Select the min samples split for the Decision Tree
    min_samples_split = st.sidebar.slider(
        '',
        min_value=2, max_value=10, value=2  
    )

    # Explanation of min_samples_leaf
    min_samples_leaf_explanation = """
    The min_samples_leaf parameter determines the minimum number of samples that must be present in a leaf node.
    - A smaller value allows the tree to create smaller leaves, which can capture more detail but may lead to overfitting.
    - A larger value creates larger leaves, which can help prevent overfitting by smoothing the model.
    """

    # Add an explanation section
    st.sidebar.write(min_samples_leaf_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select min samples leaf</div>',
        unsafe_allow_html=True
    )

    # Select the min samples leaf for the Decision Tree
    min_samples_leaf = st.sidebar.slider(
        '',
        min_value=1, max_value=10, value=1  # Default is 1
    )

    max_features_explanation = """
    The max_features parameter determines the maximum number of features considered for splitting a node.
    - None: All features are considered.
    - auto: Equivalent to 'sqrt' for classification trees.
    - sqrt: The square root of the total number of features is considered.
    - log2: The base-2 logarithm of the total number of features is considered.
    """
    st.sidebar.write(max_features_explanation, unsafe_allow_html=True)

    # Select the max features for the Decision Tree
    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select max features</div>',
        unsafe_allow_html=True
    )

    max_features = st.sidebar.selectbox(
        '',
        options=[None, 'auto', 'sqrt', 'log2'],  # Options for the max features
        index=0  # Default is None
    )

    # Train the Decision Tree model using the resampled data
    model = DecisionTreeClassifier(
        random_state=Random_State_Seed,
        max_depth=max_depth,
        criterion=criterion,
        splitter=splitter,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )
    model.fit(X_resampled, y_resampled)

    # Calculate predictions
    y_pred = model.predict(X_test)

    # Plot confusion matrix
    class_names = ["No Sale", "Sale"]
    cm = confusion_matrix(y_test, y_pred)

    # Create a DataFrame for confusion matrix with class labels
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Create a figure
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(3, 3))

    # Draw heatmap 
    sns.heatmap(cm_df, annot=True, cmap=sns.color_palette("light:#5A9_r", as_cmap=True),
                ax=ax, cbar=False, annot_kws={"size": 6}, fmt='d')

    # Set labels and title with larger font size
    plt.xlabel('Predicted Label', fontsize=6)
    plt.ylabel('True Label', fontsize=6)
    plt.title('Confusion Matrix', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # Display the plot
    st.pyplot(fig)

    # Metrik report
    # Calculate metrics
    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })

    def display_metric(name, value, explanation):
      st.markdown(f"**{name}:** {value:.2f}")
      st.markdown(f"*{explanation}*")

    st.subheader("Model Performance Metrics")

    display_metric("Accuracy", accuracy, "This is the proportion of correct predictions. The model is accurate {:.2f}% of the time.".format(accuracy * 100))
    display_metric("Precision", precision, "This is the proportion of positive predictions that were actually correct. A high precision means that the model is good at not making false positive predictions. In this case, the model is precise {:.2f}% of the time.".format(precision * 100))
    display_metric("Recall", recall, "This is the proportion of actual positive cases that were identified. A high recall means that the model is good at finding all of the relevant cases. In this case, the model has a recall of {:.2f}%.".format(recall * 100))
    display_metric("F1-Score", f1, "This is a harmonic mean of precision and recall. It is a way of combining these two metrics into a single score. A high F1-score indicates that the model is both precise and has high recall. In this case, the model has an F1-score of {:.2f}.".format(f1))

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(metrics_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[metrics_df.Metric, np.round(metrics_df.Value, 2)],
                  fill_color='lavender',
                  align='left'))
    ])

    fig.update_layout(title='Model Performance Metrics')

    st.plotly_chart(fig)

    # Plot correlation matrix
    corr = X_resampled.corr()

    Z = linkage(corr, 'ward')
    dendro = dendrogram(Z, labels=corr.columns, leaf_rotation=90, leaf_font_size=12)

    reordered_corr = corr.loc[dendro['ivl'], dendro['ivl']]

    # Display the correlation matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(reordered_corr, cmap='coolwarm', annot=False, ax=ax, cbar_kws={"shrink": .75})
    plt.title("Correlation Matrix")
    st.pyplot(fig)
           
    # Get feature importances
    feature_importances = pd.DataFrame(model.feature_importances_,
                                      index=X_train.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

    # Select top 10 features
    top_10_features = feature_importances.head(10)

    # top 10 feature importances as a table
    st.subheader("Top 10 Feature Importances")
    st.table(top_10_features)

    # Plot top 10 feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_10_features['importance'], y=top_10_features.index, palette='viridis', ax=ax)
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.set_title('Top 10 Feature Importances')
    st.pyplot(fig)



elif classifier_name == "Random Forest Classifier":
    st.subheader("Model: Random Forest Classifier")
    image_url = "https://raw.githubusercontent.com/smt07/webdatamining/main/Random%20Forest%20Classifier.png"
    st.image(image_url, caption="Random Forest Classifier", use_column_width=True)

    st.write("""The Random Forest classifier is an ensemble learning method that constructs a multitude of decision trees at training time 
    and outputs the class that is the mode of the classes of the individual trees. It is known for its robustness and ability to handle 
    large datasets with higher dimensionality.""")

    Random_State_Seed = 1

    # Resample the training data with the RandomOversampler
    strategy = 'auto'  # Default strategy
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=Random_State_Seed)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    st.write("""Resampling involves creating a new transformed version of the training 
    dataset in which the selected examples have a different class distribution.
    This is a simple and effective strategy for imbalanced classification problems.""")
    st.write("Shape of resampled dataset with onehot encoding and Random Over Sampling",
    X_resampled.shape, "Number of classes", len(np.unique(y_resampled)))

    # Select the hyperparameters for the Random Forest
    n_estimators_explanation = """
    The number of trees in the forest. More trees can improve performance but also increase computational cost.
    """
    st.sidebar.write(n_estimators_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select number of estimators</div>',
        unsafe_allow_html=True
    )

    # Select the number of estimators for the Random Forest
    n_estimators = st.sidebar.slider(
        '',
        min_value=10, max_value=200, value=100  # Default value
    )

    criterion_explanation = """
    The criterion determines the function to measure the quality of a split in the decision trees.
    - Gini: Measures the impurity of the nodes. It tends to create smaller trees.
    - Entropy: Measures the information gain based on the reduction of entropy after the split.
    """

    st.sidebar.write(criterion_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select criterion for the Random Forest</div>',
        unsafe_allow_html=True
    )

    # Select the criterion for the Random Forest
    criterion = st.sidebar.selectbox(
        '',
        options=["gini", "entropy"], 
        index=0  # Default is "gini"
    )

    max_depth_explanation = """
    The maximum depth of the trees. Higher depth allows more intricate patterns to be captured 
    but increases the risk of overfitting to the training data.
    """

    st.sidebar.write(max_depth_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select max depth of the Random Forest</div>',
        unsafe_allow_html=True
    )

    # Select the max depth of the Random Forest
    max_depth = st.sidebar.select_slider(
        '',
        options=range(1, 21),  # Assuming the depth ranges from 1 to 20
        value=5  # Default value
    )

    min_samples_split_explanation = """
    The min_samples_split parameter determines the minimum number of samples required to split an internal node. 
    - A smaller value allows the trees to consider more splits, potentially creating deeper trees.
    - A larger value can prevent overfitting by limiting the trees' growth.
    """

    st.sidebar.write(min_samples_split_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select min samples split</div>',
        unsafe_allow_html=True
    )

    # Select the min samples split for the Random Forest
    min_samples_split = st.sidebar.slider(
        '',
        min_value=2, max_value=10, value=2  # Default is 2
    )

    min_samples_leaf_explanation = """
    The min_samples_leaf parameter determines the minimum number of samples that must be present in a leaf node.
    - A smaller value allows the trees to create smaller leaves, which can capture more detail but may lead to overfitting.
    - A larger value creates larger leaves, which can help prevent overfitting by smoothing the model.
    """
    st.sidebar.write(min_samples_leaf_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select min samples leaf</div>',
        unsafe_allow_html=True
    )

    # Select the min samples leaf for the Random Forest
    min_samples_leaf = st.sidebar.slider(
        '',
        min_value=1, max_value=10, value=1  # Default is 1
    )

    max_features_explanation = """
    The max_features parameter determines the maximum number of features considered for splitting a node.
    - None: All features are considered.
    - auto: Equivalent to 'sqrt' for classification trees.
    - sqrt: The square root of the total number of features is considered.
    - log2: The base-2 logarithm of the total number of features is considered.
    """
    st.sidebar.write(max_features_explanation, unsafe_allow_html=True)

    # Select the max features for the Random Forest with custom styling
    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select max features</div>',
        unsafe_allow_html=True
    )

    max_features = st.sidebar.selectbox(
        '',
        options=[None, 'auto', 'sqrt', 'log2'], 
        index=0  
    )

    bootstrap_explanation = """
    The bootstrap parameter determines whether bootstrap samples are used when building trees.
    - True: Bootstrap samples are used.
    - False: The entire dataset is used to build each tree.
    """

    st.sidebar.write(bootstrap_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select bootstrap option</div>',
        unsafe_allow_html=True
    )

    # Select the bootstrap option for the Random Forest
    bootstrap = st.sidebar.selectbox(
        '',
        options=[True, False],  
        index=0 
    )

    # Train the Random Forest model using the resampled data
    model = RandomForestClassifier(
        random_state=Random_State_Seed,
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap
    )
    model.fit(X_resampled, y_resampled)

    # Calculate predictions
    y_pred = model.predict(X_test)

    
    # Plot confusion matrix
    class_names = ["No Sale", "Sale"]
    cm = confusion_matrix(y_test, y_pred)

    # Create a DataFrame for confusion matrix with class labels
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(3, 3))

    # Draw heatmap 
    sns.heatmap(cm_df, annot=True, cmap=sns.color_palette("light:#5A9_r", as_cmap=True),
                ax=ax, cbar=False, annot_kws={"size": 6}, fmt='d')

    # Set labels and title with larger font size
    plt.xlabel('Predicted Label', fontsize=6)
    plt.ylabel('True Label', fontsize=6)
    plt.title('Confusion Matrix', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    # Display the plot
    st.pyplot(fig)

    # Metrik report
    # Calculate metrics
    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })

    def display_metric(name, value, explanation):
      st.markdown(f"**{name}:** {value:.2f}")
      st.markdown(f"*{explanation}*")

    st.subheader("Model Performance Metrics")

    display_metric("Accuracy", accuracy, "This is the proportion of correct predictions. The model is accurate {:.2f}% of the time.".format(accuracy * 100))
    display_metric("Precision", precision, "This is the proportion of positive predictions that were actually correct. A high precision means that the model is good at not making false positive predictions. In this case, the model is precise {:.2f}% of the time.".format(precision * 100))
    display_metric("Recall", recall, "This is the proportion of actual positive cases that were identified. A high recall means that the model is good at finding all of the relevant cases. In this case, the model has a recall of {:.2f}%.".format(recall * 100))
    display_metric("F1-Score", f1, "This is a harmonic mean of precision and recall. It is a way of combining these two metrics into a single score. A high F1-score indicates that the model is both precise and has high recall. In this case, the model has an F1-score of {:.2f}.".format(f1))

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(metrics_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[metrics_df.Metric, np.round(metrics_df.Value, 2)],
                  fill_color='lavender',
                  align='left'))
    ])

    fig.update_layout(title='Model Performance Metrics')

    st.plotly_chart(fig)

    # Plot correlation matrix
    corr = X_resampled.corr()

    # Use hierarchical clustering to cluster the correlation matrix
    Z = linkage(corr, 'ward')
    dendro = dendrogram(Z, labels=corr.columns, leaf_rotation=90, leaf_font_size=12)

    # Reorder the correlation matrix based on clustering
    reordered_corr = corr.loc[dendro['ivl'], dendro['ivl']]

    # Display the clustered correlation matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(reordered_corr, cmap='coolwarm', annot=False, ax=ax, cbar_kws={"shrink": .75})
    plt.title("Correlation Matrix")
    st.pyplot(fig)
           
    # Get feature importances
    feature_importances = pd.DataFrame(model.feature_importances_,
                                      index=X_train.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

    # Select top 10 features
    top_10_features = feature_importances.head(10)

    # Display top 10 feature importances
    st.subheader("Top 10 Feature Importances")
    st.table(top_10_features)

    # Plot top 10 feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_10_features['importance'], y=top_10_features.index, palette='viridis', ax=ax)
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.set_title('Top 10 Feature Importances')
    st.pyplot(fig)

elif classifier_name == "Naive Bayes Classifier":
    st.subheader("Model: Naive Bayes Classifier")
    naive_url = "https://raw.githubusercontent.com/smt07/webdatamining/main/naivebayesclassification.webp"
    st.image(naive_url, caption="Naive Bayes Classifier", use_column_width=True)
    st.write("""The Naive Bayes classifier is a probabilistic classifier based on Bayes' theorem 
    with the assumption of independence between features. """)

    st.write(data.head())
   
    st.write("""The Naive Bayes classifier is a probabilistic classifier based on Bayes' theorem 
    with the assumption of independence between features. It is particularly effective for text 
    classification and is simple and efficient in practice.""")

    classifier_type_explanation = """
    Select the type of Naive Bayes classifier to use:
    - GaussianNB: Suitable for continuous data and assumes that the features follow a normal distribution.
    - MultinomialNB: Suitable for discrete data, particularly effective for text classification and document classification.
    - BernoulliNB: Suitable for binary/boolean features.
    """

    # Add an explanation section
    st.sidebar.write(classifier_type_explanation)

    st.sidebar.markdown(
        '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select Naive Bayes Classifier Type</div>',
        unsafe_allow_html=True
    )

    # Select the type of Naive Bayes classifier
    classifier_type = st.sidebar.selectbox(
        '',
        options=["GaussianNB", "MultinomialNB", "BernoulliNB"],  # Options for the classifier type
        index=0  # Default is GaussianNB
    )

    # Train the selected Naive Bayes model
    if classifier_type == "GaussianNB":
        model = GaussianNB()
    elif classifier_type == "MultinomialNB":
        alpha_explanation = """
        The alpha parameter for MultinomialNB is a smoothing parameter to handle zero probabilities in the data.
        """

        # Add an explanation section
        st.sidebar.write(alpha_explanation)

        st.sidebar.markdown(
            '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select alpha for MultinomialNB</div>',
            unsafe_allow_html=True
        )

        # Select the alpha parameter for MultinomialNB
        alpha = st.sidebar.slider(
            '',
            min_value=0.0, max_value=1.0, value=1.0, step=0.01  # Default is 1.0
        )
        
        model = MultinomialNB(alpha=alpha)
    elif classifier_type == "BernoulliNB":
        alpha_explanation = """
        The alpha parameter for BernoulliNB is a smoothing parameter to handle zero probabilities in the data.
        """

        # Add an explanation section
        st.sidebar.write(alpha_explanation)

        st.sidebar.markdown(
            '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select alpha for BernoulliNB</div>',
            unsafe_allow_html=True
        )

        # Select the alpha parameter for BernoulliNB
        alpha = st.sidebar.slider(
            '',
            min_value=0.0, max_value=1.0, value=1.0, step=0.01  # Default is 1.0
        )

        binarize_explanation = """
        The binarize parameter is used to binarize the data (make it binary) by thresholding.
        """

        # Add an explanation section
        st.sidebar.write(binarize_explanation)

        st.sidebar.markdown(
            '<div style="font-weight: 900; font-style: italic; color: darkblue;">Select binarize threshold for BernoulliNB</div>',
            unsafe_allow_html=True
        )
        binarize = st.sidebar.slider(
            '',
            min_value=0.0, max_value=1.0, value=0.0, step=0.01  # Default is 0.0
        )

        model = BernoulliNB(alpha=alpha, binarize=binarize)

    # Train the Naive Bayes model using the training data
    model.fit(X_train, y_train)

    # Calculate predictions
    y_pred = model.predict(X_test)

    # Plot confusion matrix
    class_names = ["No Sale", "Sale"]
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Create a figure
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(3, 3))

    # Draw heatmap 
    sns.heatmap(cm_df, annot=True, cmap=sns.color_palette("light:#5A9_r", as_cmap=True),
                ax=ax, cbar=False, annot_kws={"size": 6}, fmt='d')

    plt.xlabel('Predicted Label', fontsize=6)
    plt.ylabel('True Label', fontsize=6)
    plt.title('Confusion Matrix', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    st.pyplot(fig)

    # Metrik report
    # Calculate metrics
    accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    
    # Create DataFrame to hold the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })

    def display_metric(name, value, explanation):
      st.markdown(f"**{name}:** {value:.2f}")
      st.markdown(f"*{explanation}*")

    st.subheader("Model Performance Metrics")

    display_metric("Accuracy", accuracy, "This is the proportion of correct predictions. The model is accurate {:.2f}% of the time.".format(accuracy * 100))
    display_metric("Precision", precision, "This is the proportion of positive predictions that were actually correct. A high precision means that the model is good at not making false positive predictions. In this case, the model is precise {:.2f}% of the time.".format(precision * 100))
    display_metric("Recall", recall, "This is the proportion of actual positive cases that were identified. A high recall means that the model is good at finding all of the relevant cases. In this case, the model has a recall of {:.2f}%.".format(recall * 100))
    display_metric("F1-Score", f1, "This is a harmonic mean of precision and recall. It is a way of combining these two metrics into a single score. A high F1-score indicates that the model is both precise and has high recall. In this case, the model has an F1-score of {:.2f}.".format(f1))

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(metrics_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[metrics_df.Metric, np.round(metrics_df.Value, 2)],
                  fill_color='lavender',
                  align='left'))
    ])

    fig.update_layout(title='Model Performance Metrics')

    st.plotly_chart(fig)

    