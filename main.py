import warnings
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data import DataSets
from adaline import AdaLine

warnings.filterwarnings("ignore")


def getModel(model_name="perceptron"):
    if model_name == "perceptron":
        return Perceptron()
    if model_name == "adaline":
        return AdaLine()
    if model_name == "logistic_regression": 
        return LogisticRegression()
    if model_name == "adaboost":
        return AdaBoostClassifier()

    
def makePred(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                              random_state=42)
    for model_name in ["perceptron", 
                       "adaline", 
                       "logistic_regression", 
                       "adaboost"]:
        clf = getModel(model_name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        res = "  model: {0}, accuracy: {1:.3f}".format(model_name, accuracy)
        print(res)


def main():
    datasets = DataSets()
    for path, X, y in datasets:
        print("data set:", path)
        makePred(X, y)
    

if __name__ == "__main__":
    main()
