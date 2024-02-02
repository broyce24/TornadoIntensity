'''
Ben Royce
CSE 163 AC

This program analyzes tornado data in the US from 1950 to 2021, using machine
 learning algorithms to predict tornado intensity. This program compares three
 options (a classifier, a regressor, and a multilayer perceptron) for how well
 they perform on an ordinal classification task (predicting tornado magnitude).
'''
import pandas as pd
import numpy as np
# Classifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# Regressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
# Neural Network (Multilayer Perceptron)
from sklearn.neural_network import MLPClassifier
# Result Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def clean_data(data, cols):
    '''
    Takes in the dataframe and a list of columns.
    Chunks down the data to only include the provided columns, removes missing
     values, converts the time into a numeric value.
    Returns the cleaned dataframe.
    '''
    df = data[cols].copy()

    # Must convert time to numerical value
    def to_minutes(time):
        total = 0
        hr, min, sec = time.split(':')
        total += 60 * int(hr) + int(min)
        return total
    df['time'] = df['time'].map(to_minutes)
    # Dropping rows with unlabeled magnitude
    df = df.drop(df[df['mag'] < 0].index)
    return df


DATA = clean_data(pd.read_csv('1950-2021_all_tornadoes.csv'),
                  ["yr", "mo", "dy", "time", "slat", "slon", "len", "wid",
                   "inj", "fat", "mag"])
X = DATA.loc[:, DATA.columns != 'mag']
Y = DATA['mag']
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2)


# Standard Classifier
def classify(depth):
    '''
    Takes in a depth and trains a decision tree classifier to that depth.
    Returns the test accuracy of the model.
    '''
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_TRAIN, Y_TRAIN)

    test_pred = model.predict(X_TEST)
    test_acc = accuracy_score(Y_TEST, test_pred)
    return test_acc


# Standard Regressor with Rounding
def regress(depth):
    '''
    Takes in a depth and trains a decision tree regressor to that depth.
    Returns the test accuracy of the model, comparing against rounded values.
    '''
    model = DecisionTreeRegressor(max_depth=depth)
    model.fit(X_TRAIN, Y_TRAIN)

    test_pred = model.predict(X_TEST)
    rounded_test = np.rint(test_pred)
    test_acc = mean_squared_error(Y_TEST, rounded_test)
    return test_acc


# Neural Network
def create_nn(size, lr):
    '''
    Trains a multilayer perceptron based on the given shape and learning rate.
    Returns the test accuracy of the model.
    '''
    model = MLPClassifier(hidden_layer_sizes=size, learning_rate_init=lr,
                          max_iter=50)
    model.fit(X_TRAIN, Y_TRAIN)
    return model.score(X_TEST, Y_TEST)


def test_nn():
    '''
    Returns the optimal hyperparameters for the multilayer perceptron from a
    list of hardcoded options, and the highest test accuracy.
    '''
    sizes = [(10,), (10, 10), (10, 10, 10), (10, 10, 10, 10),
             (10, 10, 10, 10, 10)]
    lrs = [0.1, 0.001]
    # best size, learning rate, and their corresponding test accuracy
    best = 0, 0, 0
    for size in sizes:
        for lr in lrs:
            trials = [create_nn(size=size, lr=lr) for _ in range(3)]
            test_acc = sum(trials) / len(trials)
            if test_acc > best[2]:
                best = size, lr, test_acc
    return best


def test_acc(method):
    '''
    Takes in one of the decision tree strategies and returns a tuple with the
     optimal depth and highest test accuracy.
    '''
    best = (0, 1) if method == regress else (0, 0)
    for depth in range(1, 16):
        # for 3 trials
        trials = [method(depth) for _ in range(3)]
        test_acc = sum(trials) / len(trials)
        if (method == classify) and (test_acc > best[1]):
            best = depth, test_acc
        elif (method == regress) and (test_acc < best[1]):
            best = depth, test_acc
    return best


def validate(splits, method, depth, scoring):
    '''
    Cross validates our results from the decision trees using a k-fold split
     of 5.
    Returns the mean accuracy score.
    '''
    kf = KFold(n_splits=splits, shuffle=True, random_state=1)
    model = method(max_depth=depth)
    scores = cross_val_score(model, X, Y, scoring=scoring, cv=kf)
    return sum(scores) / len(scores)


def main():
    opt_class = test_acc(classify)
    opt_reg = test_acc(regress)
    opt_nn = test_nn()
    print(opt_class)
    print(opt_reg)
    print(opt_nn)

    # Perform validation of the decision trees
    classifier = validate(5, DecisionTreeClassifier, depth=9,
                          scoring='accuracy')
    regressor = -1 * validate(5, DecisionTreeRegressor, depth=9,
                              scoring='neg_mean_squared_error')
    print(classifier)
    print(regressor)


if __name__ == '__main__':
    main()
