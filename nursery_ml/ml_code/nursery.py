import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    GradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

# LOADING THE DATA
data = pd.read_csv("/Users/user/PycharmProjects/classification_ml/nursery/nursery_ml/data/nursery.data", sep=",")

pd.set_option('display.max_columns', 9)
print(data.head())

# DATA CLEANING
data.columns = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "admission"]
print(data)

columns_unique_values = data.apply(lambda x: x.unique())
print(columns_unique_values)
print(data.has_nurs.unique())
print(data.admission.unique())

# It can be seen that the has_nurs column has a unique value "complete" and "completed" which are essentially the same
# they both mean that the form was completely filled. Hence, completed will be replaced with complete.
data.form = data.form.map(lambda x:  "complete" if x == "completed" else x)

print(data.apply(lambda x: x.unique()))

# The admission column is ordinal
data.admission = data.admission.apply(
    lambda dp:
    0 if dp == "not_recom"
    else 1 if dp == "recommend"
    else 2 if dp == "very_recom"
    else 3 if dp == "priority"
    else 4  # special priority
)

print(data.admission.value_counts())

# Now looking at the data again, there is need to collapse some categories, to
# allow for a balance of categories
# Hence, recommend, very_recom and priority may be collapsed to recommend_ws_pr i.e.
# "recommended without special priority"
# Hence the categories will be "not recommended" , "recommended without special priority" and "not recommended"

data.admission = data.admission.apply(
    lambda dp:
    dp if dp == 0 # Not Recommended
    else 1 if dp in [1, 2, 3]  # Recommended without special priority
    else 2  # Recommended with special priority
)

print(data.admission.value_counts())

data_ = data.copy(deep=True)  # a copy of data for visualization
print("AA", len(data.admission))

print("AA", data_.index)
# print(data_.value_counts())

plot = sns.barplot(x=data_.admission.unique(), y=data_.admission.value_counts())
plot.set(xticklabels=["Not Recommended", "No Special Priority ", "Special Priority"])
plt.title("Bar Chart of Admission Status")
plt.xlabel("Admission Status")
plt.ylabel("Count")
plt.savefig('/Users/user/PycharmProjects/classification_ml/nursery/nursery_images/bar_plot_of_admission_status.jpeg', format="jpeg")
plt.show()

target_class_count = data.admission.value_counts()
print("\nThe percentage of data points")
print(100 * (target_class_count/(data.admission.value_counts().sum())).round(decimals=4))

# The other columns need to be encoded.
"""encoder = LabelEncoder()
data.iloc[:, 0:-2] = data.iloc[:, 0:-2].apply(lambda x: encoder.fit_transform(x))
print(data)"""

data.parents = data.parents.apply(
    lambda p: 0 if p == "usual"
    else 1 if p == "pretentious"
    else 2  # i.e p == great_pret  # greatly pretentious
)

data.has_nurs = data.has_nurs.apply(
    lambda h: 0 if h == "proper"
    else 1 if h == "less_proper"
    else 2 if h == "improper"  # i.e p == great_pret (greatly pretentious)
    else 3 if h == "critical"
    else 4  # i.e  very critical
)

data.form = data.form.apply(
    lambda f: 0 if f == "complete"
    else 1 if f == "incomplete"
    else 2  # i.e. f == complete
)

data.children = data.children.apply(
    lambda f: 4 if f == "more"
    else f  # i.e f in [1, 2, 3]
)

data.housing = data.housing.apply(
    lambda h: 0 if h == "convenient"
    else 1 if h == "less_conv"  # i.e less convenient
    else 2  # i.e. f == critical
)

data.finance = data.finance.apply(
    lambda f: 0 if f == "convenient"
    else 1  # i.e inconvenient
)

data.social = data.social.apply(
    lambda s: 0 if str(s) == "nonprob"
    else 1 if str(s) == "slightly_prob"  # i.e slightly problematic
    else 2  # h = problematic
)
data.health = data.health.apply(
    lambda h: 0 if h == "priority"
    else 1 if h == "not_recom"  # i.e. not recommended
    else 2  # h = recommended
)


# FEATURE SELECTION
k_list = []  # each k stands for the number of features to be included in the model
accuracy_list = []  # the corresponding accuracy for each k
y = data.admission
x = data.drop(labels=["admission"], axis=1)


def features_selection(features, target, no_of_features):
    selector = SelectKBest(k=no_of_features)
    selector.fit_transform(features, target)
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})
    print("The score of each feature:", selector.scores_.round(decimals=2))

    selected_x = selector.get_feature_names_out(input_features=None)
    # print(selected_x)
    print("The selected features are", selected_x)
    x_data_frame = features.loc[:, selected_x]
    return x_data_frame


# Now the k with the maximum precision is used to build the model.
x = features_selection(x, y, 6)

print("Value Counts")
print(x.value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=123)

# lists needed to display the accuracies of the models and save them
joblib_file_names = [
                "RandomForest.joblib",
                "ExtraTrees.joblib",
                "Bagging_KNN_Model.joblib",
                "GradientBoostingClassifier.joblib"
]  # A list of the filenames of the models to be serialized by joblib.

models = []  # A list of the model objects to be serialized by joblib

models_names = [
    "Random Forest",
    "Extremely Randomized Trees",
    "Bagging for K-Nearest Neighbour",
    "Stochastic Gradient Tree Boosting"
]

out_of_bag_scores = []
out_of_bag_errors = []
test_accuracies = []
estimators = []  # the model with the recommended parameters after tuning


# To print the model outputs
def print_model_results(model_name_, oob_score_, oob_error_, accuracy_, y_predicted):
    print("_"*100)
    print(model_name_.upper())
    print("_"*100)
    print("Out of bag score = ", round(oob_score_, 4) if type(oob_score_) == np.float64 else oob_score_)
    print("Out of bag error = ", round(oob_error_, 4) if type(oob_error_) == np.float64 else oob_error_)
    print("Test Accuracy = ", round(accuracy_, 4))
    print("\n \t\t\t\t\t", "Classification Report")
    print(classification_report(y_predicted, y_test))


# 1) Random Forest
# By trial and error, I have chosen these parameters which give me an out of bag error of 0.1904. Holding
# all other parameters constant, a maximum depth of 8 gives the lowest out of bag error
random_forest_model = RandomForestClassifier(
    n_estimators=300,
    oob_score=True,
    random_state=123,
    criterion='entropy',  # {'gini', 'log_loss', 'entropy'}
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=20,
    max_features="sqrt",  # max_features : {"sqrt", "log2", None}
    max_leaf_nodes=65
)

random_forest_model.fit(x_train, y_train)
# print(best_rf_model.estimator)

test_accuracy = random_forest_model.score(x_test, y_test)
out_of_bag_score = random_forest_model.oob_score_
out_of_bag_error = 1 - random_forest_model.oob_score_

# A dataframe of the predicted output variable y
y_pred = random_forest_model.predict(x_test)

# Outputs for the random forest model
print_model_results("random forest", out_of_bag_score, out_of_bag_error, test_accuracy, y_pred)

# Appending the metrics to the appropriate lists
test_accuracies.append(test_accuracy)
out_of_bag_scores.append(out_of_bag_score)
out_of_bag_errors.append(out_of_bag_error)
estimators.append(random_forest_model)


# 2) Extremely Randomized Trees
# By trial and error, I have chosen these parameters. Holding
# all other parameters constant, a maximum depth of 8 gives the lowest out of bag error
extra_trees = ExtraTreesClassifier(
    n_estimators=300,
    oob_score=True,
    bootstrap=True,
    random_state=123,
    criterion='entropy',  # {'gini', 'log_loss', 'entropy'}
    max_depth=9,
    min_samples_split=30,
    min_samples_leaf=30,
    max_features="sqrt",
    max_leaf_nodes=65
)


# Tuning the logistic model to find the best model parameters which make up the best estimator
"""print("Searching for the best LOR model ...")
clf = GridSearchCV(logistic_model, parameters)
clf.fit(x_train, y_train)"""

# best estimator for the logistic regression
# best_lor_estimator = clf.best_estimator_


# Fitting the selected (best) model
extra_trees.fit(x_train, y_train)

test_accuracy = extra_trees.score(x_test, y_test)
out_of_bag_score = extra_trees.oob_score_
out_of_bag_error = 1 - extra_trees.oob_score_

# A dataframe of the predicted output variable y
y_pred = extra_trees.predict(x_test)

# Output for the Extremely Randomized Trees Model
print_model_results("extremely randomized trees", out_of_bag_score, out_of_bag_error, test_accuracy, y_pred)

# Appending the metrics to the appropriate lists
test_accuracies.append(test_accuracy)
out_of_bag_scores.append(out_of_bag_score)
out_of_bag_errors.append(out_of_bag_error)
estimators.append(extra_trees)


# 3) Bagging Classifier for K-Nearest Neighbors
knn_model = KNeighborsClassifier()

# Searching for the best model
"""parameters = dict(
    n_neighbors=range(1, 25, 2),
    leaf_size=[10, 20, 30, 40, 50, 60, 70, 80, 100]
)

clf = GridSearchCV(estimator=knn_model, param_grid=parameters)
clf.fit(x_train, y_train)

best_knn_estimator = clf.best_estimator_"""

best_knn_estimator = KNeighborsClassifier(leaf_size=40, n_neighbors=23)
# print("best knn estimator:", best_knn_estimator)

bagging_knn_model = BaggingClassifier(
    estimator=best_knn_estimator,
    n_estimators=400,
    max_features=6,
    bootstrap=True,
    bootstrap_features=True,
    oob_score=True,
    random_state=123
)

bagging_knn_model.fit(x_train, y_train)

test_accuracy = bagging_knn_model.score(x_test, y_test)
out_of_bag_score = bagging_knn_model.oob_score_
out_of_bag_error = 1 - bagging_knn_model.oob_score_

# A dataframe of the predicted output variable y
y_pred = bagging_knn_model.predict(x_test)

# Output for the bagging-KNN-estimator
print_model_results("bagging for k-nearest Neighbour", out_of_bag_score, out_of_bag_error, test_accuracy, y_pred)

# Appending the metrics to the appropriate lists
test_accuracies.append(test_accuracy)
out_of_bag_scores.append(out_of_bag_score)
out_of_bag_errors.append(out_of_bag_error)
estimators.append(bagging_knn_model)


# 4) Stochastic Gradient Tree Boosting
gradient_boost_model = GradientBoostingClassifier(
    loss='log_loss',
    learning_rate=1,
    n_estimators=300,
    subsample=1.0,
    min_samples_split=30,
    min_samples_leaf=30,
    max_depth=8,
    random_state=123,
    max_features=6,
)

gradient_boost_model.fit(x_train, y_train)

test_accuracy = gradient_boost_model.score(x_test, y_test)
out_of_bag_score = extra_trees.oob_score_
out_of_bag_error = 1 - extra_trees.oob_score_

# Output for the Voting Classifier
print_model_results("stochastic gradient tree boosting", out_of_bag_score, out_of_bag_error, test_accuracy, y_pred)

# A dataframe of the predicted output variable y
y_pred = gradient_boost_model.predict(x_test)

# Appending the metrics to the appropriate lists
test_accuracies.append(test_accuracy)
out_of_bag_scores.append(out_of_bag_score)
out_of_bag_errors.append(out_of_bag_error)
estimators.append(gradient_boost_model)


# The data frame for comparing the performances of the models
print("_"*100)
print("Summary of the Models")
print("_"*100)
models_and_accuracies = pd.DataFrame({
    "Model Name": models_names,
    "Estimator": estimators,
    "Out of Bag Score": out_of_bag_scores,
    "Out of Bag Error": out_of_bag_errors,
    "Test Accuracy": test_accuracies
})

models_and_accuracies.iloc[:, [2, 3, 4]] = models_and_accuracies.iloc[:, [2, 3, 4]].apply(lambda fig: round(fig, 6))

print(models_and_accuracies)


# Best Model according to test score
m_and_a = models_and_accuracies
for i in range(len(models_and_accuracies)):
    if m_and_a.iloc[i, 4] == m_and_a.iloc[:, 4].max():
        print("Best Model According Test Accuracy is",
              m_and_a.iloc[i, 0],
              "With an accuracy of",
              m_and_a.iloc[i, 4])


# 7) saving the model
for i in range(len(estimators)):
    joblib.dump(estimators[i],
                "/Users/user/PycharmProjects/classification_ml/nursery/nursery_ml/models/" + joblib_file_names[i],
                compress=5 if i == 2 else 3)

# The best model is used to predict whether workers will be given to absenteeism or not
file = "/Users/user/PycharmProjects/classification_ml/nursery/nursery_ml/models/Bagging_KNN_Model.joblib"
bagging_knn_model = joblib.load(file)
y_pred = bagging_knn_model.predict(x_test)
y_ypred = pd.DataFrame(np.array([y_test, y_pred])).T
y_ypred.columns = ["Original y", "Predicted y"]

y_ypred["Original y"] = y_ypred["Original y"].map(
    lambda fx:
    "Not Recommended" if fx == 0
    else "Recommended without Special Priority" if fx == 1
    else "Recommended with Special Priority"
)

y_ypred["Predicted y"] = y_ypred["Predicted y"].map(
    lambda fx:
    "Not Recommended" if fx == 0
    else "Recommended without Special Priority" if fx == 1
    else "Recommended with Special Priority"
)

y_ypred.to_csv(
    "/Users/user/PycharmProjects/classification_ml/nursery/nursery_ml/prediction/"
    + "classification_result.csv"
)
