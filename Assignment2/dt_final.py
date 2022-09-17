import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.naive_bayes import (GaussianNB, BernoulliNB)
import matplotlib.pyplot as plt
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor, AdaBoostRegressor)
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, AdaBoostClassifier)
from sklearn.metrics import mean_squared_error
# SMOTE (Synthetic Minority Oversampling Technique)
from imblearn.over_sampling import SMOTE
from collections import Counter



#__________________________________________________________K NEAREST NEIGHBORS__________________________________________________________

def my_knn(train, test, k=21):
    
#     print("column names Train \n", train.columns.tolist())
#     print("\n\nfeature set size : ", len(train.columns.tolist()))
#     print("column names Test \n", test.columns.tolist())
#     print("\n\nfeature set size : ", len(test.columns.tolist()))
    print("value of k = ",k)
    labels = train['credit']

#     print(labels)

    features = train.drop(['credit'],axis = 1)
#     features_cols = features.columns

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)

    print("\n\ntraining set size after train_test_split = ", len(X_train))
    print("\ntest set size = ",len(y_test))
    
#____________SMOTE BLOCK_________________    
    # summarize class distribution
    print("Before oversampling: ",Counter(y_train))

    # define oversampling strategy
    smote = SMOTE()

    # fit and apply the transform
    X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

    # summarize class distribution
    print("After oversampling: ",Counter(y_train_SMOTE))


    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred = knn.predict(X_test)
#     y_pred = knn.predict(test)
#     accuracy_nums(y_pred,y_test)
    return y_test, y_pred
#     return y_pred


def optimizingk(train, test):
    f1results = []
    K = []
    f1=None
    for k in range(30):
        K.append(2*k+3)
        y_test, y_pred = my_knn(train,test,2*k+3)
        f1 = getf1(y_pred, y_test)
        f1results.append(f1)
    optk = 2*(np.argmax(f1results)) + 3
    max_f1 = np.amax(f1results)
    print("optimum k = ",optk)
    print("max F1 Score = ", max_f1)
    plt.figure(figsize=(16,8))
    plt.plot(K, f1results, 'bx-')
    plt.xlabel('k')
    plt.ylabel('F1 Score')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
#_____________________________________________________KNN END__________________________________________________________





#__________________________________________________________TREEE OPERATIONS__________________________________________________________

def d_tree(train, test):

#     cat_var = ['F3', 'F4', 'F7', 'F8', 'F9','F10', 'F11']

#     train_dummies = pd.get_dummies(train,columns = cat_var)
#     test_dummies = pd.get_dummies(test,columns = cat_var)

#     print("column names Train \n", train_dummies.columns.tolist())
#     print("\n\nfeature set size : ", len(train_dummies.columns.tolist()))
#     print("column names Test \n", test_dummies.columns.tolist())
#     print("\n\nfeature set size : ", len(test_dummies.columns.tolist()))

    labels = train['credit']

    # print(labels)

    features = train.drop(['credit'],axis = 1)

    # print(features)


    # def check_skewness(column_name):
    # column_name = 'F11'
    # column_name = 'F10'

    # sns.countplot(x = column_name, data=train, palette='hls')
    # plt.xlabel(column_name, fontsize=12)
    # plt.ylabel('count', fontsize=12)
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)
    
    #____________SMOTE BLOCK_________________    
    # summarize class distribution
    print("Before oversampling: ",Counter(y_train))

    # define oversampling strategy
    smote = SMOTE()

    # fit and apply the transform
    X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

    # summarize class distribution
    print("After oversampling: ",Counter(y_train_SMOTE))

#     print("\n\ntraining set size after train_test_split = ", len(X_train))
#     print("\ntest set size = ",len(y_test))

    gini_tree = DecisionTreeClassifier(criterion = "gini", splitter='best', max_depth = 12)
    
    gini_tree.fit(X_train_SMOTE,y_train_SMOTE)
#     gini_tree.fit(features,labels)
#     y_pred = gini_tree.predict(X_test)
    y_pred = gini_tree.predict(X_test)
    accuracy_nums(y_pred,y_test)
#     str_arr = np.array_str(y_pred,max_line_width=1)
#     print(len(y_pred))
#     print(y_pred)

#     return y_test, y_pred
    return y_pred
    
def getoptdepth(train, test):
    
    labels = train['credit']
    features = train.drop(['credit'],axis = 1)
    
    gini_f1 = []
    ent_f1 = []
    index = []
    gini_tree = None
    ent_tree = None
    y_pred_gini = None
    y_pred_ent = None
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.50)
    
    for i in range(50):
        index.append(i)
        if i == 0:
            gini_tree = DecisionTreeClassifier(criterion = "gini", splitter='best')
            gini_tree.fit(X_train,y_train)
            y_pred_gini = gini_tree.predict(X_test)
            gini_f1.append(getf1(y_pred_gini, y_test))
            
            ent_tree = DecisionTreeClassifier(criterion = "entropy", splitter='best')
            ent_tree.fit(X_train,y_train)
            y_pred_ent = ent_tree.predict(X_test)
            ent_f1.append(getf1(y_pred_ent, y_test))
        else:
            gini_tree = DecisionTreeClassifier(criterion = "gini", splitter='best', max_depth = i)
            gini_tree.fit(X_train,y_train)
            y_pred_gini = gini_tree.predict(X_test)
            gini_f1.append(getf1(y_pred_gini, y_test))
            
            ent_tree = DecisionTreeClassifier(criterion = "entropy", splitter='best', max_depth = i)
            ent_tree.fit(X_train,y_train)
            y_pred_ent = ent_tree.predict(X_test)
            ent_f1.append(getf1(y_pred_ent, y_test))
    
    opt_depth_ent = np.argmax(ent_f1)
    opt_depth_gini = np.argmax(gini_f1)
    max_f1_gini = np.amax(gini_f1)
    max_f1_ent = np.amax(ent_f1)
    
    print("\noptimal deth for gini tree = {} and its F1 : {}".format(opt_depth_gini, max_f1_gini))
    print("\noptimal deth for entropy tree = {} and its F1 : {}".format(opt_depth_ent, max_f1_ent))
    
    plt.figure(figsize=(8, 8))
    plt.title('Decision Tree depth VS F1 Score')
    plt.plot(index, gini_f1, 'b-', color="red", label='Gini Index')
    plt.plot(index, ent_f1, 'b-', color="blue", label='Entropy')
    plt.legend(loc='upper right')
    plt.xlabel('Index')
    plt.ylabel('F1_score')
    plt.show()
    

#__________________________________________________________TREEE OPERATIONS END__________________________________________________________
    

#__________________________________________________________FEATURE IMPORTANCE BLOCK__________________________________________________________
    
def d_tree_FI(train,test):

#     after feature importancd drop f11_1, f10_1, f7, f9, f8
# #     train = train.drop(columns=['F11_1'])
#     train = train.drop(columns=['F9'])
#     train = train.drop(columns=['F7'])
#     train = train.drop(columns=['F10_1'])
#     train = train.drop(columns=['F6'])
#     train = train.drop(columns=['F8'])
    
#     after feature importancd drop f11_1, f10_1, f7, f9, f8
#     test = test.drop(columns=['F11_1'])
#     test = test.drop(columns=['F9'])
#     test = test.drop(columns=['F7'])
#     test = test.drop(columns=['F10_1'])
#     test = test.drop(columns=['F6'])
#     test = test.drop(columns=['F8'])

#     cat_var = ['F3', 'F4', 'F7', 'F8', 'F9','F10', 'F11']

#     train_dummies = pd.get_dummies(train,columns = cat_var)
#     test_dummies = pd.get_dummies(test,columns = cat_var)

    print("column names Train \n", train.columns.tolist())
    print("\n\nfeature set size : ", len(train.columns.tolist()))
    print("column names Test \n", test.columns.tolist())
    print("\n\nfeature set size : ", len(test.columns.tolist()))

    labels = train['credit']

    # print(labels)

    features = train.drop(['credit'],axis = 1)
    features_cols = features.columns

    # print(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)

#     print("\n\ntraining set size after train_test_split = ", len(X_train))
#     print("\ntest set size = ",len(y_test))

    gini_tree = DecisionTreeClassifier(criterion = "gini", splitter='best', max_depth = 10)
    gini_tree.fit(X_train,y_train)
#     gini_tree.fit(features,labels)
#     y_pred = gini_tree.predict(X_test)
    y_pred = gini_tree.predict(X_test)
    # str_arr = np.array_str(y_pred,max_line_width=1)
 
    # feature Importance printed and then the lowest valued features are removed and tested for accuracy
    
    feat_importance = gini_tree.tree_.compute_feature_importances(normalize=False)
    feat_imp_dict = dict(zip(features_cols, gini_tree.feature_importances_))
    feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
    feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
    print(feat_imp.sort_values(by=['FeatureImportance'], ascending=False))
#     return y_test, y_pred    
    return ypred

#__________________________________________________________FEATURE IMPORTANCE BLOCK END__________________________________________________________




#__________________________________________________________IMPLEMENTING DIFFERENT NAIVE BAYES CLASSIFIERS__________________________________________________________

def gnbc(train, test):

    cat_var = ['F3', 'F4', 'F7', 'F8', 'F9','F10_1', 'F11_1']

    train_dummies = pd.get_dummies(train,columns = cat_var)
    test_dummies = pd.get_dummies(test,columns = cat_var)

#     print("column names Train \n", train_dummies.columns.tolist())
#     print("\n\nfeature set size : ", len(train_dummies.columns.tolist()))
#     print("column names Test \n", test_dummies.columns.tolist())
#     print("\n\nfeature set size : ", len(test_dummies.columns.tolist()))


#     labels = train['credit']
#     print(labels)
#     features = train.drop(['credit'],axis = 1)
#     print("features:\n\n", features.head)

    labels = train_dummies['credit']
#     print(labels)
    features = train_dummies.drop(['credit'],axis = 1)
#     print("features:\n\n", features.head)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)
    
    # summarize class distribution
    print("Before oversampling: ",Counter(y_train))

    # define oversampling strategy
    smote = SMOTE()

    # fit and apply the transform
    X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

    # summarize class distribution
    print("After oversampling: ",Counter(y_train_SMOTE))

#     # print("\n\ntraining set size after train_test_split = ", len(X_train))
#     # print("\ntest set size = ",len(y_test))
    
    #Create a Gaussian Classifier
    gnb = GaussianNB()

    #Train the model using the training sets
    gnb.fit(X_train_SMOTE, y_train_SMOTE)
#     gnb.fit(features, labels)

    #Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    accuracy_nums(y_pred,y_test)
    y_pred = gnb.predict(test_dummies)
    return y_pred    
#     return y_pred

#defining bernoulli's naive bayes

def bnbc(train, test):
    cat_var = ['F3', 'F4', 'F7', 'F8', 'F9','F10_1', 'F11_1']

    train_dummies = pd.get_dummies(train,columns = cat_var)
    test_dummies = pd.get_dummies(test,columns = cat_var)

#     print("column names Train \n", train_dummies.columns.tolist())
#     print("\n\nfeature set size : ", len(train_dummies.columns.tolist()))
#     print("column names Test \n", test_dummies.columns.tolist())
#     print("\n\nfeature set size : ", len(test_dummies.columns.tolist()))
    
    labels = train_dummies['credit']
    print(labels)
    features = train_dummies.drop(['credit'],axis = 1)
    print("features:\n\n", features.head)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)
    
    bgnb = BernoulliNB()

    #Train the model using the training sets
    bgnb.fit(X_train, y_train)
#     gnb.fit(features, labels)

    #Predict the response for test dataset
#     y_pred = bgnb.predict(X_test)
    y_pred = bgnb.predict(test_dummies)
#     return y_test, y_pred    
    return y_pred
        
#__________________________________________________________NAIVE BAYES END__________________________________________________________


#_______________________________________________________HELPER FUNCTIONS______________________________________________
def accuracy_nums(y_pred,y_test):
    
    print("\n\nsize of prediction: ", len(y_pred))

    print("\nConfusion Matrix: ", confusion_matrix(y_test, y_pred))

    print ("\nAccuracy : ", accuracy_score(y_test,y_pred)*100) 

    print("\nReport : ", classification_report(y_test, y_pred))
  
    print("\nF1 Score: ", f1_score(y_test, y_pred))
    
def getf1(y_pred, y_test):
    
    return f1_score(y_test, y_pred)


#_______________________________________________________HELPER BLOCK END________________________________________________


#__________________________________________BOOTSTRAPPING AND RANDOM FOREST EXPERIMENTS___________________________________

def bootstrapping(train,test):
    
#     one-hot encoding block

    cat_var = ['F3', 'F4', 'F7', 'F8', 'F9','F10_1', 'F11_1']

    train_dummies = pd.get_dummies(train,columns = cat_var)
    test_dummies = pd.get_dummies(test,columns = cat_var)

#     print("column names Train \n", train_dummies.columns.tolist())
#     print("\n\nfeature set size : ", len(train_dummies.columns.tolist()))
#     print("column names Test \n", test_dummies.columns.tolist())
#     print("\n\nfeature set size : ", len(test_dummies.columns.tolist()))
    
    labels = train_dummies['credit']
#     print(labels)
    features = train_dummies.drop(['credit'],axis = 1)
#     print("features:\n\n", features.head)
    
#     one-hot encoding block end


#     storing the parameters and creating the future value(f1_s and iterations(estimators)) arrays
    random_state = 42
    n_jobs = 1  # Parallelisation factor for bagging, random forests
    n_estimators = 1000
    step_factor = 10
    axis_step = int(n_estimators/step_factor)
#     axis_step = 100
    
#     Use the training-testing split with 70% of data in the training data with the remaining 30% of data in the testing
#     print("column names Train \n", train.columns.tolist())
#     print("\n\nfeature set size : ", len(train.columns.tolist()))
#     print("column names Test \n", test.columns.tolist())
#     print("\n\nfeature set size : ", len(test.columns.tolist()))

#     labels = train['credit']
# #     print(labels)
#     features = train.drop(['credit'],axis = 1)
# #     features_cols = features.columns

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)
    
#     Pre-create the arrays which will contain the f1_s for each particular ensemble method
    estimators = np.zeros(axis_step)
    bagging_f1_s = np.zeros(axis_step)
    rf_f1_s = np.zeros(axis_step)
    boosting_f1_s = np.zeros(axis_step)
    bag_pred = None
    boost_pred = None
    rf_pred = None
    
    dt = DecisionTreeClassifier()
    
    print("X_train : {}\n shape {}".format(X_train.columns.tolist(),X_train.shape))
    print("X_test : {}\n shape {}".format(X_test.columns.tolist(),X_test.shape))
    
#     Estimate the Bagging f1_s over the full number of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Bagging Estimator: %d of %d..." % (step_factor*(i+1), n_estimators))
        bagging = BaggingClassifier(dt, n_estimators=step_factor*(i+1), n_jobs=n_jobs, random_state=random_state, bootstrap = True)
        bagging.fit(X_train, y_train)
        bag_pred = bagging.predict(X_test)
        f1_s = f1_score(y_test, bag_pred)
        estimators[i] = step_factor*(i+1)
        bagging_f1_s[i] = f1_s
        
    bagging_max = np.amax(bagging_f1_s)
    bagging_max_index = np.argmax(bagging_f1_s)
    
#     Estimate the Random Forest f1_s over the full number of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Random Forest Estimator: %d of %d..." % (step_factor*(i+1), n_estimators))
        rf = RandomForestClassifier(n_estimators=step_factor*(i+1), n_jobs=n_jobs, random_state=random_state, bootstrap = True)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        f1_s = f1_score(y_test, rf_pred)
        estimators[i] = step_factor*(i+1)
        rf_f1_s[i] = f1_s
    
    rf_max = np.amax(rf_f1_s)
    rf_max_index = np.argmax(rf_f1_s)
    
#     Estimate the AdaBoost f1_s over the full number of estimators, across a step size ("step_factor")
#     for i in range(0, axis_step):
#         print("Boosting Estimator: %d of %d..." % (step_factor*(i+1), n_estimators))
#         boosting = AdaBoostClassifier(dt, n_estimators=step_factor*(i+1), random_state=random_state, learning_rate=0.01)
#         boosting.fit(X_train, y_train)
#         boost_pred = boosting.predict(X_test)
#         f1_s = f1_score(y_test, boost_pred)
#         estimators[i] = step_factor*(i+1)
#         boosting_f1_s[i] = f1_s
     
#     boosting_max = np.amax(boosting_f1_s)
#     boosting_max_index = np.argmax(boosting_f1_s)
    
    print("\n\nbagging max F1 Score : {} at {} confirming val {}".format(bagging_max, bagging_max_index, bagging_f1_s[bagging_max_index]))
    print("\nRandom Forest max F1 Score : {} at {} confirming val {}".format(rf_max, rf_max_index, rf_f1_s[rf_max_index]))
#     print("\nBoosting max F1 Score : {} at {} confirming val {}".format(boosting_max, boosting_max_index, boosting_f1_s[boosting_max_index]))
    
#     Plot the chart of f1_s versus number of estimators
    plt.figure(figsize=(8, 8))
    plt.title('Bagging and Random Forest comparison')
    plt.plot(estimators, bagging_f1_s, 'b-', color="red", label='Bagging')
    plt.plot(estimators, rf_f1_s, 'b-', color="blue", label='Random Forest')
#     plt.plot(estimators, boosting_f1_s, 'b-', color="black", label='AdaBoost')
    plt.legend(loc='upper right')
    plt.xlabel('Estimators')
    plt.ylabel('F1 Score')
    plt.show()
    
def predrf(train,test):
    
#     storing the parameters and creating the future value(f1_s and iterations(estimators)) arrays
    random_state = 42
    n_jobs = 1  # Parallelisation factor for bagging, random forests
    n_estimators = 470
    step_factor = 10
    max_depth = 27
#     axis_step = int(n_estimators/step_factor)
#     axis_step = 100
    
#     Use the training-testing split with 70% of data in the training data with the remaining 30% of data in the testing
    print("column names Train \n", train.columns.tolist())
    print("\n\nfeature set size : ", len(train.columns.tolist()))
    print("column names Test \n", test.columns.tolist())
    print("\n\nfeature set size : ", len(test.columns.tolist()))

    labels = train['credit']
#     print(labels)
    features = train.drop(['credit'],axis = 1)
#     features_cols = features.columns

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)
    
    #____________SMOTE BLOCK_________________    
    # summarize class distribution
    print("Before oversampling: ",Counter(y_train))

    # define oversampling strategy
    smote = SMOTE()

    # fit and apply the transform
    X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

    # summarize class distribution
    print("After oversampling: ",Counter(y_train_SMOTE))
        
    rf = RandomForestClassifier(n_estimators = n_estimators, n_jobs=n_jobs, random_state=random_state, max_depth=35, min_samples_split=41)
#     rf = RandomForestClassifier(max_depth = 27, min_samples_split=38)
    rf.fit(X_train_SMOTE, y_train_SMOTE)
    rf_pred = rf.predict(X_test)
    accuracy_nums(rf_pred,y_test)
    rf_pred = rf.predict(test)
    return rf_pred 
    
    
def getrfdepth(train,test):
    
    labels = train['credit']
    features = train.drop(['credit'],axis = 1)
    
    rf_f1 = []
    index = []
    rf = None
    rf_pred = None
    row = None
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)
    
    for i in range(10, 40):
        row = []
        for j in range(3, 51, 5):
            print("{}  {}".format(i,j))
            if i == 0:
                rf = RandomForestClassifier(min_samples_split=j)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                row.append(f1_score(y_test,rf_pred))
            else:
                rf = RandomForestClassifier(max_depth = i, min_samples_split=j)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                row.append(f1_score(y_test,rf_pred))
        rf_f1.append(row)
        
    opt_depth = np.argmax(rf_f1)
    max_f1 = np.amax(rf_f1)
    
    print("\noptimal depth and sample_split for random forest = {} and its F1 : {}".format(opt_depth, max_f1))
    
#     plt.figure(figsize=(8, 8))
#     plt.title('Random Forest Depth VS F1 Score')
#     plt.plot(index, rf_f1, 'b-', color="red", label='rf')
# #     plt.legend(loc='upper right')
#     plt.xlabel('Index')
#     plt.ylabel('F1_score')
#     plt.show()
    
    
#__________________________________________________________BOOTSRAPPING AND RF END__________________________________________________________


    
#_________________________________________________________RUNNER FUNCTION____________________________________________________________    
    
def runner():
    train_file = "1613441452_005314_1600106342_882043_train.csv"
    test_file = "1613441452_0172963_1600106342_8864183_test.csv"
    
#     create dataframes form the data and drop the id column
    train = pd.read_csv(train_file, sep=',', header = 0)
    train = train.drop(columns=['id'])
    test = pd.read_csv(test_file, sep=',', header = 0)
    test = test.drop(columns=['id'])
    
#    label encoding for F10 and F11 features
    train["F10"] = train["F10"].astype('category')
#     print(train.dtypes)
    train["F10_1"] = train["F10"].cat.codes
#     print(train.head())
    train["F11"] = train["F11"].astype('category')
#     print(train.dtypes)
    train["F11_1"] = train["F11"].cat.codes
#     print(train.head())
    train = train.drop(columns=['F10'])
    train = train.drop(columns=['F11'])
    
    test["F10"] = test["F10"].astype('category')
#     print(test.dtypes)
    test["F10_1"] = test["F10"].cat.codes
#     print(test.head())  
    test["F11"] = test["F11"].astype('category')
#     print(test.dtypes)
    test["F11_1"] = test["F11"].cat.codes
#     print(test.head())
    test = test.drop(columns=['F10'])
    test = test.drop(columns=['F11'])

#     optimizingk(train, test)
#     y_pred = my_knn(train, test, 21)
#     y_pred = gnbc(train, test)
    y_pred = predrf(train, test)
#     y_pred = d_tree(train,test)
#     getrfdepth(train, test)
#     bootstrapping(train, test)
#     accuracy_nums(y_pred,y_test)
#   Write the prediction in class

    f1 = open("RF_SMOTE_d35_split41.txt", "w")
    for i in y_pred:
#         print(str(i))
        f1.write(str(i))
        f1.write(os.linesep)
    f1.close()

    
    
runner()

