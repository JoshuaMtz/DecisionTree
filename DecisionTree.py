# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import pandas as pd
import seaborn as sns
import random
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from IPython.display import Image  
from matplotlib import pyplot as plt
from pprint import pprint
from six import StringIO
from sklearn import metrics
from sklearn import preprocessing
from statistics import mode

"""# Train and Test data"""

def Split_train_test(dataFrame, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(dataFrame))

    indices = dataFrame.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_dataFrame = dataFrame.loc[test_indices]
    train_dataFrame = dataFrame.drop(test_indices)
    return train_dataFrame, test_dataFrame

"""# Data purity"""

def Purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    if len(unique_classes) == 1:
        return True
    else:
        return False

"""# Classification"""

def Classify(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

"""# Potential Splits"""

def Potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  #The last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits

"""# Split Data"""

def Split(data, split_column, split_value):
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    #Continuous data 
    if type_of_feature == "continuous":
        dBelow = data[split_column_values <= split_value]
        dAbove = data[split_column_values >  split_value]
    
    #Categorical data  
    else:
        dBelow = data[split_column_values == split_value]
        dAbove = data[split_column_values != split_value]
    
    return dBelow, dAbove

"""# Lowest Overall Entropy"""

def Entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

def Overall_entropy(dBelow, dAbove):
    n = len(dBelow) + len(dAbove)
    if n != 0:
      p_dBelow = len(dBelow) / n
      p_dAbove = len(dAbove) / n
    else:
      p_dBelow = 0
      p_dAbove = 0

    overall_entropy =  (p_dBelow * Entropy(dBelow) 
                      + p_dAbove * Entropy(dAbove))
    
    return overall_entropy

def Best_split(data, potential_splits):
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            dBelow, dAbove = Split(data, split_column=column_index, split_value=value)
            current_overall_entropy = Overall_entropy(dBelow, dAbove)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

"""# Type of Feature"""

def Type_of_Feature(dataFrame):
    feature_types = []
    n_unique_values_treshold = 15
    for feature in dataFrame.columns:
        if feature != "label":
            unique_values = dataFrame[feature].unique()
            individual_value = unique_values[0]

            if (isinstance(individual_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

"""# Algorithm"""

def Decision_tree(dataFrame, contador=0, min_samples=2, max_depth=5):
    #Preparation of data
    if contador == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = dataFrame.columns
        FEATURE_TYPES = Type_of_Feature(dataFrame)
        data = dataFrame.values
    else:
        data = dataFrame           
        
    var=Purity(data)
    #print("Hola: ",str(var))
    #Base cases
    if (Purity(data)) or (len(data) < min_samples) or (contador == max_depth):
        classification = Classify(data)
        return classification

    
    # recursive part
    else:    
        contador += 1

        #Splits 
        potential_splits = Potential_splits(data)
        split_column, split_value = Best_split(data, potential_splits)
        dBelow, dAbove = Split(data, split_column, split_value)
        
        #Analyse for empty data
        if len(dBelow) == 0 or len(dAbove) == 0:
            classification = Classify(data)
            return classification
        
        #Determine questions for categorical and continuous data
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        #Continuous
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        #Categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        #Sub-tree
        sub_tree = {question: []}
        
        #Answers (recursion)
        yes_answer = Decision_tree(dBelow, contador, min_samples, max_depth)
        no_answer = Decision_tree(dAbove, contador, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree

"""# Classification"""

def Classify_individual(individual, tree):
    question = list(tree.keys())[0]
    feature_name, operator_comparision, value = question.split(" ")

    #Ask question for continuous
    if operator_comparision == "<=":
        if individual[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    #Feature is categorical
    else:
        if str(individual[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    #Recursive
    else:
        last_tree = answer
        return Classify_individual(individual, last_tree)

"""# Accuracy"""

def Accuracy(dataFrame, tree):
    pprint(tree)
    #res = dict()
    #for sub in tree:
    # split() for key 
    # packing value list
    #  key, *val = sub.split()
    #  res[key] = val
    dataFrame["classification"] = dataFrame.apply(Classify_individual, axis=1, args=(tree,))
    dataFrame["classification_correct"] = dataFrame["classification"] == dataFrame["label"]
    
    accuracy = dataFrame["classification_correct"].mean()
    
    return accuracy

"""# Load and Prepare Data"""

dataFrame = pd.read_csv("tratamientos.csv")
dataFrame["label"] = dataFrame.Drug
dataFrame = dataFrame.drop(["Drug"], axis=1)
dataFrame.head()

"""# Decision Tree"""

random.seed()

train_dataFrame, test_dataFrame = Split_train_test(dataFrame, 20)
individual = test_dataFrame.iloc[0]
print(individual)
tree = Decision_tree(train_dataFrame, max_depth=3)
accuracy = Accuracy(test_dataFrame, tree)

pprint(tree, width=50)
test_dataFrame.head()
print(accuracy)

def main():
    # load dataset
    data = pd.read_csv("tratamientos.csv", delimiter=",")
    X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    y = data["Drug"]

    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F','M'])
    X[:,1] = le_sex.transform(X[:,1]) 

    le_BP = preprocessing.LabelEncoder()
    le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
    X[:,2] = le_BP.transform(X[:,2])

    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit([ 'NORMAL', 'HIGH'])
    X[:,3] = le_Chol.transform(X[:,3]) 

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    #Create a Decision Tree Classifier Object
    model = DecisionTreeClassifier()

    #Train the Tree
    model = model.fit(x_train,y_train)

    #Predict the response for the dataset
    y_pred = model.predict(x_test)

    #Accuracy using metrics
    acc1 = metrics.accuracy_score(y_test,y_pred)*100
    print("Accuracy of the model is " + str(acc1))

    #Classification Report
    report = classification_report(y_test, y_pred)
    print(report)

    #Plotting of the tree
    features = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['0','1','2','3','4'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #graph.write_png('tratamiento_set_1.png') #Save the imahe
    #Image(graph.create_png())

    # Create Decision Tree classifer object
    model = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Train Decision Tree Classifer
    model = model.fit(x_train,y_train)

    #Predict the response for test dataset
    y_pred = model.predict(x_test)

    print(x_test)
    print("Prefictions of the model: " + str(y_pred))

    # Model Accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

    #Better Decision Tree Visualisation
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,filled=True, rounded=True,special_characters=True, feature_names = features,class_names=['0','1','2','3','4'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    #graph.write_png('tratamiento_set_2.png') #Save de image
    #Image(graph.create_png())

if __name__ == "__main__":
    main()
