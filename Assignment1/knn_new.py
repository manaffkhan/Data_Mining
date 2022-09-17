#input data in usable form
#preprocess the data
#tf-idf vectorization
#cosine similarity
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize as tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine2
from scipy.spatial.distance import cosine
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
nltk.download('genesis')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
genesis_ic = wn.ic(genesis, False, 0.0)
import numpy as np
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt



#preprocessing block

# make all text lowercase
def text_lowercase(text):
    return text.lower()

# remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# tokenize
def tokenize(text):
    text = word_tokenize(text)
    return text

# remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text

# lemmatize
lemmatizer = nltk.wordnet.WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

def preprocessing(text):
    text = text_lowercase(text)
#     print("\n\n lowercase : \n\n")
#     print(text)
#     text = remove_urls(text)
#     print("\n\n removed URLs : \n\n")
#     print(text)
    text = remove_numbers(text)
#     print("\n\n remove numbers : \n\n")
#     print(text)
    text = remove_punctuation(text)
#     print("\n\n remove punctuations : \n\n")
#     print(text)
    text = tokenize(text)
#     print("\n\n tokenize : \n\n")
#     print(text)
    text = remove_stopwords(text)
#     print("\n\n remove stopwords : \n\n")
#     print(text)
    text = lemmatize(text)
#     print("\n\n lemmatize : \n\n")
#     print(text)
    text = ' '.join(text)
    return text

# #preprocessing block end


#Running the algo:        
def knn(df_train, df_test, k):

    df_train_arr = []
    df_test_arr = []
    global_arr = []
    
    tst_index = 0
    for rev in df_train['review']:
#         print(preprocessing(str(rev)))
        df_train_arr.append(preprocessing(str(rev)))
        tst_index += 1
    
#     print(tst_index)
    global_arr = df_train_arr
    
    
    for rev in df_test['review']:
#         print(preprocessing(str(rev)))
        df_test_arr.append(preprocessing(str(rev)))
    for line in df_test_arr:
        global_arr.append(line)
        
#     print("final\n\n\n\n")
#     print(df_train_arr)
    print(tst_index)
    print(len(global_arr))
    tf_idf = TfidfVectorizer().fit_transform(global_arr)
    print(len(tf_idf.todense()))
#     print("\nto dense()\n")
#     print(tf_idf.todense())

    f1 = open('final_result.dat', 'w')
    
    print("\n\n\nENTERING SIMILARITY LOOP\n\n\n")
    cosine_op = cosine2(tf_idf[tst_index:].todense(),tf_idf[:tst_index].todense())
    print("rows")
    print(len(cosine_op))
    print
    print("columns")
    print(len(cosine_op[0]))
    
    
    for tst in range(len(cosine_op)): 
#         print(tst-tst_index)
#         print("\t")
        mean_sim = []
        for i in range(k):
            row = []
            for j in range(2):
                row.append(j-1)
            mean_sim.append(row)
        
        index = 0
        sum_class = 0
        for trn in range(tst_index):
            key = [cosine_op[tst][trn],trn]
#             print(key)
            for x in range(k):
                #key = mean_similarity_score
                if mean_sim[x][0] < key[0]:
                    temp = key
                    key = mean_sim[x]
                    mean_sim[x] = temp
                    
        for x in range(k):
            sum_class += int(df_train.at[mean_sim[x][1],'output'])
#            print(df_train.at[mean_sim[x][1],'output'])
            
        if sum_class > 0 :
            print("+1")
            f1.write("+1" + os.linesep)
        else:
            print("-1")
            f1.write("-1" + os.linesep)
#     li = []
#     num = []
#     for tst in range(len(cosine_op)):
#         li = []
#         num = []
#         sum_class=0
#         for trn in range(len(cosine_op[0])):
#             num.append(trn)
#             li.append(cosine_op[0][tst])
        
#         df_tmp = pd.DataFrame(li, columns = ['col1'])
# #         df_test = pd.read_csv(testing_file, names = ['review'])
# #         df_tmp['Index'] = np.arange(df_test.shape[0])
#         df_tmp['col2'] = num
# #         df_tmp = df_tmp.sort_values(by='col1', ascending=False)
#         df_tmp2 = df_tmp.nlargest(k, 'col1')
#         print(df_tmp2)
    
#         for i in range(k):
#             print(df_train.at[df_tmp.at[i,'col2'],'output'])
# #         if sum_class > 0 :
# #             print("+1")
# #             f1.write("+1" + os.linesep)
# #         else:
# #             print("-1")
# #             f1.write("-1" + os.linesep)
            
    f1.close()
        
    
#     for tst in range(tst_index,len(global_arr)):
#         print(tst-tst_index)
#         print("\t")
#         mean_sim = []
#         for i in range(k):
#             row = []
#             for j in range(2):
#                 row.append(j-1)
#             mean_sim.append(row)
        
#         index = 0
#         sum_class = 0
        
#         for trn in range(tst_index):
#             key = [(cosine2(tf_idf[tst].todense(),tf_idf[trn].todense())),trn]
# #             print(key)
#             for x in range(k):
#                 #key = mean_similarity_score
#                 if mean_sim[x][0] < key[0]:
#                     temp = key
#                     key = mean_sim[x]
#                     mean_sim[x] = temp
#             index += 1
# #             print(mean_sim)
#         for x in range(k):
#             sum_class += (df_train.at[mean_sim[x][1],'output'])
            
#         if sum_class > 0 :
#             print("+1")
#             f1.write("+1" + os.linesep)
#         else:
#             print("-1")
#             f1.write("-1" + os.linesep)
    
    
# knn("test_train1.dat", "test_test1.dat", 1)
#knn("1611766549_6456213_train_file.dat", "1611766549_7170458_test.dat", 7)


#OPTIMIZING THE VALUE OF K
#block start

def optimize_k(train_file):
    
    df = pd.read_csv(train_file, delimiter = '\t', names = ['output', 'review'])

    #shuffling and splitting data into 80:20
    df_80 = df.sample(frac = 0.80)
    df_20 = df.drop(df_80.index)
    sum_correct = []
    sum_c = 0
    k = 0
    for z in range(1):
        print("inside z loop with z value:")
        print(z)
        knn(df_80, df_20, 7)
#         #sum_correct = []
#         l=[]
#         f1 = open("final_result.dat", 'r')
#         for line in f1:
#             l.append(line.strip())
#         df_op = pd.DataFrame(l, columns = ['output'])
#         #df_test = pd.read_csv(testing_file, names = ['review'])
#         df_op['Index'] = np.arange(df_op.shape[0])
#         sum_c = 0
#         for i in range(df_op.count+1):
#             if df_op.at[i,'output'] == df_20.at[i,'output']:
#                 sum_c += 1
#         sum_correct.append(sum_c/df_op.shape[0])
        
#         print(2*z + 3)
#         print(sum_correct[z])
        
#     plt.figure(figsize=(10,6))
#     plt.plot(range(1,40),sum_correct,color=’blue’, linestyle=’dashed’, marker=’o’, markerfacecolor=’red’, markersize=10)
#     plt.title(‘Efficiency vs. K Value’)
#     plt.xlabel(‘K’)
#     plt.ylabel(‘Efficiency’)
    
    
        
    
def runalgo(train_file, test_file):
    #create dataframe from the training .dat file
    df_train = pd.read_csv(train_file, delimiter = '\t', names = ['output', 'review'])
    df_train['Index'] = np.arange(df_train.shape[0])
    
    l=[]
    test_file = open(test_file, 'r')
    for line in test_file:
        l.append(line.strip())
    df_test = pd.DataFrame(l, columns = ['review'])
    #df_test = pd.read_csv(testing_file, names = ['review'])
    df_test['Index'] = np.arange(df_test.shape[0])
    k = 25
    knn(df_train, df_test, k)
    
    
# optimize_k("1611766549_6456213_train_file.dat")
runalgo("1611766549_6456213_train_file.dat", "1611766549_7170458_test.dat")