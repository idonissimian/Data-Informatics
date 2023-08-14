# Name: Ido Nissimian
# ID: 206835282

import pandas as pd
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import copy


# Constants
DATA_LEN = 1000


# Loading the data
def data_loading():
    # Get data from excel file
    facebook_data = pd.read_excel(r'Facebook_Data.xlsx')
    # Change columns names to numbers
    data = facebook_data.drop(['Name'], axis=1)
    data.columns = np.arange(len(data.columns))
    return data
    

# Generate gender and age for each element in data
def generate_gender_and_age():
    gender_and_age_df = pd.DataFrame(index=range(DATA_LEN))
    # Gender with random probability: 1 - male, 2 - female
    np.random.seed(0)
    gender_and_age_df['gender'] = np.random.randint(1, 3, DATA_LEN)
    # Get n and p for var = 25 and std = 5 
    n = 100
    p = 0.26
    # var = binom.var(n, p) # For checking
    # std = binom.std(n, p) # For checking
    gender_and_age_df['age'] = np.random.binomial(n, p, 1000)
    return gender_and_age_df, n


# Plot gender histogram 
def plot_gender_hist(gender_and_age_df):
    # Creating the dataset
    male_sum = gender_and_age_df['gender'].to_list().count(1)
    female_sum = DATA_LEN - male_sum
    data_for_hist = {'male': male_sum, 'female': female_sum}      
    # Creating the bar plot
    x = list(data_for_hist.keys())
    y = list(data_for_hist.values())
    fig, ax = plt.subplots()
    # Plotting the graph 
    bar_x_y = ax.bar(x, y)
    ax.set(title='Gender Histogram')
    ax.bar_label(bar_x_y)
    
    
# Plot age histogram 
def plot_age_hist(gender_and_age_df, n):
    # List of distribution values
    age_list = gender_and_age_df['age'].to_list()
    # Plotting the graph 
    plt.hist(age_list)
    plt.show()
        

# Create and plot graph (contains 100 firat nodes)
def create_and_plot_graph_100_first_nodes(data, gender_and_age_df):
    # Get only 100 first node (for visualization)
    net_graph_100 = nx.from_pandas_adjacency(data.iloc[:100,:100])  
    color_map = []
    gender_list_100 = gender_and_age_df['gender'].to_list()[:100]
    for gender in gender_list_100:
        if gender == 1:
            color_map.append('blue')
        else: 
            color_map.append('red')      
    nx.draw(net_graph_100, node_color=color_map, with_labels = True)
    
    
# Calculate and plot parameters of network
def calculate_parameters_of_network(net_graph):
    # Clustering coefficient
    clustering_dict = nx.clustering(net_graph)
    clustering_avg = nx.average_clustering(net_graph)
    # plt.hist(clustering_dict.values())
    # plt.show()
    # Shortest path
    shortest_path = dict(nx.all_pairs_shortest_path(net_graph))
    # Create histogram of the len of the shortest paths
    shortest_path_len_list = []
    for k1 in shortest_path.keys():
        for k2 in shortest_path[k1].keys():
            if k1 != k2:
                shortest_path_len_list.append(len(shortest_path[k1][k2]))
    shortest_path_len_avg = sum(shortest_path_len_list)/len(shortest_path_len_list)
    path2 = 0
    path3 = 0
    for path_len in shortest_path_len_list:
        if path_len == 2:
            path2 += 1
        if path_len == 3:
            path3 += 1
    fig, ax = plt.subplots()
    # Plotting the graph 
    bar_x_y = ax.bar(['2','3'], [path2,path3])
    ax.set(title='Shortest Path Histogram')
    ax.bar_label(bar_x_y)
    
    
# Compute assortativity
def plot_assortativity(net_graph):
    y_degree_list = nx.degree_histogram(net_graph)
    x_degree_list = list(range(len(y_degree_list)))
    plt.bar(x_degree_list, y_degree_list, align='center')
    plt.xlabel('degree')
    plt.ylabel('frequency of degree')
    plt.show()
    
    # assort = list(nx.node_degree_xy(net_graph, x="in", y="out"))
    # y_degree_degree_list = list(zip(*assort))[1]
    # x_degree_degree_list = list(zip(*assort))[0]
    # plt.bar(x_degree_degree_list, y_degree_degree_list, align='center')
    # plt.xlabel('degree')
    # plt.ylabel('degree of degree')
    # plt.show()
    
    # x_list = []
    # y_list = []
    # for node1 in net_graph.nodes:
    #     node1_neighbors = net_graph.neighbors(node1)
    #     node1_degree = net_graph.degree[node1]
    #     x_list.append(node1_degree)
    #     node1_sum_len_neighbors = 0
    #     for node2 in node1_neighbors: 
    #         node1_sum_len_neighbors += net_graph.degree[node2]
    #     y_list.append(round(node1_sum_len_neighbors/node1_degree, 2))
    # plt.bar(x_list, y_list, align='center')
    # plt.xlabel('neighbors')
    # plt.ylabel('neighbors of neighbors')
    # plt.show()
       
# Calculate local clustering coefficient
def local_clustering_coefficient(gender_and_age_df):
    gender_and_age_df['age_group'] = pd.qcut(gender_and_age_df['age'], q=5, labels=range(1,6))
    return gender_and_age_df
    

# Remove males and females
def remove_males_and_females(data, gender_and_age_df):
    # remove males
    data_without_males = data
    gender_list = gender_and_age_df['gender'].to_list()
    for i in range(data.shape[0]):
        if gender_list[i] == 1: #male
            data_without_males = data_without_males.drop([i], axis=0)
            data_without_males = data_without_males.drop([i], axis=1)
    # Plot 100 first node
    net_graph_without_males = nx.from_pandas_adjacency(data_without_males) 
    net_graph_without_males_100 = nx.from_pandas_adjacency(data_without_males.iloc[:100,:100])  
    nx.draw(net_graph_without_males_100, node_color=['red'], with_labels = True)
    clustering_avg_without_males = nx.average_clustering(net_graph_without_males)
    # remove females
    data_without_females = data
    gender_list = gender_and_age_df['gender'].to_list()
    for i in range(data.shape[0]):
        if gender_list[i] == 2: # female
            data_without_females = data_without_females.drop([i], axis=0)
            data_without_females = data_without_females.drop([i], axis=1)
    # Plot 100 first node
    net_graph_without_females = nx.from_pandas_adjacency(data_without_females) 
    net_graph_without_females_100 = nx.from_pandas_adjacency(data_without_females.iloc[:100,:100])  
    nx.draw(net_graph_without_females_100, with_labels = True)
    clustering_avg_without_females = nx.average_clustering(net_graph_without_females)        
    # Calculate average shortest path for each graph
    shortest_path_len_avg_list = []
    for g in [net_graph_without_males, net_graph_without_females]:
        shortest_path = dict(nx.all_pairs_shortest_path(g))
        # Create histogram of the len of the shortest paths
        shortest_path_len_list = []
        for k1 in shortest_path.keys():
            for k2 in shortest_path[k1].keys():
                if k1 != k2:
                    shortest_path_len_list.append(len(shortest_path[k1][k2]))
        shortest_path_len_avg = sum(shortest_path_len_list)/len(shortest_path_len_list)
        shortest_path_len_avg_list.append(shortest_path_len_avg)
    return clustering_avg_without_males, clustering_avg_without_females, shortest_path_len_avg_list[0], shortest_path_len_avg_list[1]
          

def create_bias_in_data(data, net_graph):
    gender_list_bias = []
    for node1 in net_graph.nodes:
        node1_degree = net_graph.degree[node1]
        # If degree less than 90 - male, else - female
        if node1_degree <= 90:
            gender_list_bias.append(1) # male
        else:
            gender_list_bias.append(2) # female
    gender_and_age_bias_df = pd.DataFrame(index=range(DATA_LEN))
    gender_and_age_bias_df['gender'] = pd.Series(gender_list_bias)
    n = 100
    p = 0.26
    gender_and_age_bias_df['age'] = np.random.binomial(n, p, 1000)
    # plot_gender_hist(gender_and_age_bias_df)
    # clustering_avg_without_males_bias, clustering_avg_without_females_bias, shortest_path_len_avg_without_males_bias, shortest_path_len_avg_without_females_bias = remove_males_and_females(data, gender_and_age_bias_df)
    return gender_and_age_bias_df
    

def prepare_data_for_model(data, net_graph, gender_and_age_bias_df):
    data_gender = pd.DataFrame(1000*[gender_and_age_bias_df['gender'].to_list()])
    data_for_model = data.mul(data_gender)
    data_for_model['target'] = gender_and_age_bias_df['gender']
    return data_for_model
    

def random_forest_classifier_train_all_data(data_for_model):
    # Train all over the data
    X_train = data_for_model.drop(columns=['target'])
    y_train = data_for_model['target']
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    # Predict of 10%: 100 nodes
    _, X_test, _, y_test = train_test_split(X_train, y_train, test_size=0.1)
    y_pred = rf.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    font = {'weight' : 'bold',
    'size'   : 20}
    plt.rc('font', **font)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2]).plot()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    

def random_forest_classifier(data_for_model):
    # Train 90% of the data
    X = data_for_model.drop(columns=['target'])
    y = data_for_model['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    rf_clf = RandomForestClassifier(random_state=137)
    rf_clf.fit(X_train, y_train)
    # Predict of 10%: 100 nodes
    y_pred = rf_clf.predict(X_test)
    # Calculate metrics
    print("random forest classifier:")
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    font = {'weight' : 'bold',
    'size'   : 20}
    plt.rc('font', **font)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2]).plot()
    print("accuracy:", accuracy_score(y_test, y_pred)) 
    print("precision:", precision_score(y_test, y_pred))
    print("recall:", recall_score(y_test, y_pred))
    print("TP:", cm[1][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("TN:", cm[0][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FP:" , cm[0][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FN:", cm[1][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    

def gradient_boosting_classifier(data_for_model):
    # Train 90% of the data
    X = data_for_model.drop(columns=['target'])
    y = data_for_model['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X_train, y_train)
    # Predict of 10%: 100 nodes
    y_pred = gb_clf.predict(X_test)
    # Calculate metrics
    print("gradient boosting classifier:")
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    font = {'weight' : 'bold',
    'size'   : 20}
    plt.rc('font', **font)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2]).plot()
    print("accuracy:", accuracy_score(y_test, y_pred)) 
    print("precision:", precision_score(y_test, y_pred))
    print("recall:", recall_score(y_test, y_pred))
    print("TP:", cm[1][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("TN:", cm[0][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FP:" , cm[0][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FN:", cm[1][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))


def svm_classifier(data_for_model):
    # Train 90% of the data
    X = data_for_model.drop(columns=['target'])
    y = data_for_model['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    svm_clf = svm.SVC()
    svm_clf.fit(X_train, y_train)
    # Predict of 10%: 100 nodes
    y_pred = svm_clf.predict(X_test)
    print("SVM classifier:")
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    font = {'weight' : 'bold',
    'size'   : 20}
    plt.rc('font', **font)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2]).plot()
    print("accuracy:", accuracy_score(y_test, y_pred)) 
    print("precision:", precision_score(y_test, y_pred))
    print("recall:", recall_score(y_test, y_pred))
    print("TP:", cm[1][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("TN:", cm[0][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FP:" , cm[0][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FN:", cm[1][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    

# Create heavy tail using degree tranformation
def design_heavy_tailed_distributions(data, net_graph):
    new_data = copy.deepcopy(data)
    new_net_graph = copy.deepcopy(net_graph)
    nodes_list = list(net_graph.nodes)
    
    # # Run over all nodes
    # for i in range(len(nodes_list)):
    #     for j in range(len(nodes_list)):
    #         # Remove 2 nodes and append to node with more neighboors
    #         if i != j and new_net_graph.degree[j] > new_net_graph.degree[i]:
    #             for k in range(len(nodes_list)):
    #                 flag = 0
    #                 if new_data.iloc[i, k] == 1:
    #                     new_data.iloc[i, k] = 0
    #                     flag += 1
    #                 if flag == 2:
    #                     break
    #             for t in range(len(nodes_list)):
    #                 flag1 = 0
    #                 if new_data.iloc[j, t] == 0:
    #                     new_data.iloc[j, t] = 1
    #                     flag1 += 1
    #                 if flag1 == 2:
    #                     break    
    #             new_net_graph = nx.from_pandas_adjacency(new_data)
    #             break

    # Create intial network with barabasi model - 250,000 edges
    new_net_graph = nx.barabasi_albert_graph(len(nodes_list), 500)
    new_data = nx.to_pandas_adjacency(new_net_graph)
    new_net_graph = nx.from_pandas_adjacency(new_data) 
    # Add 315 edges
    flag = 0
    for i in range(len(nodes_list)):
        for j in range(len(nodes_list)):
            if flag == 315:
                break
            if new_data.iloc[i, j] == 0:
                new_data.iloc[i, j] = 1
                flag += 1
    new_net_graph = nx.from_pandas_adjacency(new_data)
    # Plot degree
    y_degree_list = nx.degree_histogram(new_net_graph)
    x_degree_list = list(range(len(y_degree_list)))
    plt.clf()
    plt.bar(x_degree_list, y_degree_list, align='center')
    plt.xlabel('degree')
    plt.ylabel('frequency of degree')
    plt.show()
    return new_data, new_net_graph
    

def split_bias_data_for_model(data_for_model, gender_and_age_bias_df):
    # Test contains only males
    data_for_model_sorted = data_for_model.sort_values(by=['target'])
    X_y_train_bias = data_for_model_sorted.iloc[100:,]
    X_y_test_bias = data_for_model_sorted.iloc[:100,]
    X_train_bias = X_y_train_bias.drop(columns=['target'])
    y_train_bias = X_y_train_bias['target']
    X_test_bias = X_y_test_bias.drop(columns=['target'])
    y_test_bias = X_y_test_bias['target']
    return X_train_bias, X_test_bias, y_train_bias, y_test_bias


def random_forest_classifier_bias_data(X_train_bias, X_test_bias, y_train_bias, y_test_bias):
    rf_clf_bias = RandomForestClassifier()
    rf_clf_bias.fit(X_train_bias, y_train_bias)
    y_pred_bias = rf_clf_bias.predict(X_test_bias)
    # Calculate metrics
    print("random forest classifier bias data:")
    # Create confusion matrix
    cm = confusion_matrix(y_test_bias, y_pred_bias)
    font = {'weight' : 'bold',
    'size'   : 20}
    plt.rc('font', **font)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2]).plot()
    print("accuracy:", accuracy_score(y_test_bias, y_pred_bias)) 
    print("precision:", precision_score(y_test_bias, y_pred_bias))
    print("recall:", recall_score(y_test_bias, y_pred_bias))
    print("TP:", cm[1][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("TN:", cm[0][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FP:" , cm[0][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FN:", cm[1][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])) 
    
   
def gradient_boosting_classifier_bias_data(X_train_bias, X_test_bias, y_train_bias, y_test_bias):
    gb_clf_bias = GradientBoostingClassifier()
    gb_clf_bias.fit(X_train_bias, y_train_bias)
    y_pred_bias = gb_clf_bias.predict(X_test_bias)
    # Calculate metrics
    print("gradient boosting classifier bias data:")
    # Create confusion matrix
    cm = confusion_matrix(y_test_bias, y_pred_bias)
    font = {'weight' : 'bold',
    'size'   : 20}
    plt.rc('font', **font)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2]).plot()
    print("accuracy:", accuracy_score(y_test_bias, y_pred_bias)) 
    print("precision:", precision_score(y_test_bias, y_pred_bias))
    print("recall:", recall_score(y_test_bias, y_pred_bias))
    print("TP:", cm[1][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("TN:", cm[0][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FP:" , cm[0][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FN:", cm[1][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])) 
    

def svm_classifier_bias_data(X_train_bias, X_test_bias, y_train_bias, y_test_bias):
    svm_clf_bias = svm.SVC()
    svm_clf_bias.fit(X_train_bias, y_train_bias)
    y_pred_bias = svm_clf_bias.predict(X_test_bias)
    # Calculate metrics
    print("svm classifier bias data:")
    # Create confusion matrix
    cm = confusion_matrix(y_test_bias, y_pred_bias)
    font = {'weight' : 'bold',
    'size'   : 20}
    plt.rc('font', **font)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2]).plot()
    print("accuracy:", accuracy_score(y_test_bias, y_pred_bias)) 
    print("precision:", precision_score(y_test_bias, y_pred_bias))
    print("recall:", recall_score(y_test_bias, y_pred_bias))
    print("TP:", cm[1][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("TN:", cm[0][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FP:" , cm[0][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    print("FN:", cm[1][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])) 
    
    
def main():
    data = data_loading()
    gender_and_age_df, n = generate_gender_and_age()
    plot_gender_hist(gender_and_age_df)
    plot_age_hist(gender_and_age_df, n)
    create_and_plot_graph_100_first_nodes(data, gender_and_age_df)
    net_graph = nx.from_pandas_adjacency(data) 
    gender_and_age_bias_df = create_bias_in_data(data, net_graph)
    data_for_model = prepare_data_for_model(data, net_graph, gender_and_age_bias_df)
    random_forest_classifier_train_all_data(data_for_model)
    random_forest_classifier(data_for_model)
    gradient_boosting_classifier(data_for_model)
    svm_classifier(data_for_model)
    # new_data, new_net_graph = design_heavy_tailed_distributions(data, net_graph)
    # gender_and_age_bias_df = create_bias_in_data(data, net_graph)
    
    
if __name__ == "__main__":
    main()
    

