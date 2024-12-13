import pandas as pd
import random
from gptcache.similarity_evaluation import OnnxModelEvaluation
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np

from sklearn import svm, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
#import matplotlib.pyplot as plt

random.seed(42)

onnx = OnnxModelEvaluation()
thres = 0.8

def greedy_ways(attack, thres):
    sentence_group = [attack[0]]
    for j in range(1, len(attack)):
        new_sentence = attack[j]
        
        flag = 0
        
        for vic in sentence_group:
            score = onnx.evaluation(
                {
                    'question': new_sentence
                },
                {
                    'question': vic
                }
            )
            # HIT at least for half of the victim
            # Then we consider it is HIT
            if score > thres:
                flag = 1
        
        if flag == 1:
            continue
        else:
            sentence_group.append(new_sentence)


    return sentence_group # this is the reduced attack sentences, so we only have one from each sentence space

def sample_part(list1, list2, n):

    zipped = list(zip(list1, list2))

    sampled = random.sample(zipped, n)

    unzipped_list1, unzipped_list2 = zip(*sampled)

    unzipped_list1 = list(unzipped_list1)
    unzipped_list2 = list(unzipped_list2)

    return unzipped_list1, unzipped_list2

def create_plot(same_answer, response_time):
    data = {
        "boolean_list": same_answer,
        "seconds_list": response_time,
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 7))
    for i in range(len(df)):
        color = 'blue' if df['boolean_list'][i] else 'red'
        plt.scatter(i, df['seconds_list'][i], color=color)

    handles = [plt.Line2D([0], [0], color='blue', marker='o', linestyle='', label='Semantically similar query'),
            plt.Line2D([0], [0], color='red', marker='o', linestyle='', label='Semantically different query')]
    plt.legend(handles=handles)
    plt.xlabel('Attack Query Number')
    plt.ylabel('Query Response Time')
    plt.title('Attack Queries against Response Times')

    # Save the plot as a PNG file
    plt.savefig("sameAnswer_responseTime.png", dpi=50, bbox_inches='tight')


    return

def create_plot_2(same_answer, response_time):
    # Example data
    data = {
        "boolean_list": same_answer,
        "seconds_list": response_time,
    }
    df = pd.DataFrame(data)

    # Separate the data based on the boolean_list
    false_data = df[df['boolean_list'] == False].reset_index()
    true_data = df[df['boolean_list'] == True].reset_index()

    # Adjust x-coordinates to separate the groups
    false_x = range(len(false_data))  # Sequential indices for False samples
    true_x = range(len(false_data), len(false_data) + len(true_data))  # Sequential indices for True samples

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(false_x, false_data['seconds_list'], color='red', label='False')
    plt.scatter(true_x, true_data['seconds_list'], color='blue', label='True')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Seconds')
    plt.title('Scatter Plot with False on Left and True on Right')

    # Add the legend
    plt.legend(title="Boolean Value", loc="upper right")
    plt.savefig("sameAnswer_responseTime2.png", dpi=50, bbox_inches='tight')

def pick_50(boolean_list, number_list):
    # Step 1: Zip the lists together
    zipped = list(zip(boolean_list, number_list))

    # Step 2: Separate the pairs based on boolean values
    true_pairs = [pair for pair in zipped if pair[0] == True]
    false_pairs = [pair for pair in zipped if pair[0] == False]

    # Step 3: Randomly sample 50 pairs from each group
    sampled_true = random.sample(true_pairs, 50) if len(true_pairs) >= 50 else true_pairs
    sampled_false = random.sample(false_pairs, 50) if len(false_pairs) >= 50 else false_pairs

    # Step 4: Combine the sampled pairs back together
    final_pairs = sampled_true + sampled_false

    # Step 5: Unzip the sampled pairs back into separate lists
    final_booleans, final_numbers = zip(*final_pairs)

    # Convert them back to lists (optional)
    final_booleans = list(final_booleans)
    final_numbers = list(final_numbers)

    # Print the result
    #print("Final booleans:", final_booleans)
    #print("Final numbers:", final_numbers)

    return final_booleans, final_numbers

def cm_all(tp, fn, fp, tn):
    # Construct the confusion matrix
    confusion_matrix = np.array([[tp, fp], [fn, tn]])

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Cache Hit', 'Predicted Cache Miss'], yticklabels=['Actual Cache Hit', 'Actual Cache Miss'])

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("confusion_matrix_first_attack_sentence.png", dpi=300)


    return

def cacheAttackData():
    df = pd.read_csv('./CacheHitMiss.csv')

    same_answer = df['Same Answer'].to_list()
    response_time = df['Query Response Time'].to_list()
    label_type = df['Label Type'].to_list()
    attack_sentence_num = df['Attack Sentence Number'].to_list()

    tp_count = sum(1 for b, f in zip(label_type, response_time) if b and float(f) < 1.0)
    fn_count = sum(1 for b, f in zip(label_type, response_time) if b and float(f) > 1.0)
    fp_count = sum(1 for b, f in zip(label_type, response_time) if not b and float(f) < 1.0)
    tn_count = sum(1 for b, f in zip(label_type, response_time) if not b and float(f) > 1.0)

    for (i, l) in enumerate(label_type):
        if l == True and float(response_time[i]) > 1.0:
            print(float(response_time[i]))

    print(tp_count, fp_count, fn_count)

    tpr = tp_count / label_type.count(True)
    print(tpr)
    fpr = fp_count / label_type.count(False)
    print(fpr)

    fnr = fn_count / label_type.count(True)
    print(fn_count, label_type.count(True))
    tnr = tn_count / label_type.count(False)

    filename = './semantic_datasets/data_16'
    fr = open(filename, 'rb')
    try:
        False_label = pickle.load(fr) # name and med cond are different
        #print('false label:', False_label)
    except EOFError:
        print('error')
        
    True_label = pickle.load(fr) # name and med cond are same, rest is semantically similar
    #print('true label:', True_label)
    topk_seed = pickle.load(fr)
    #print('topk_seed:', topk_seed)

    attack_sentences = greedy_ways(topk_seed, thres)

    print(True_label[0])
    print(attack_sentences[0])
    print(attack_sentences[2])

    print("tpr:", tpr)
    print("fnr:", fnr)
    print("tnr:", tnr)
    print("fpr:", fpr)

    tp_count_attack0 = sum(1 for b, f, s in zip(label_type, response_time, attack_sentence_num) if b and float(f) < 1.0 and s == 0)
    fn_count_attack0 = sum(1 for b, f, s in zip(label_type, response_time, attack_sentence_num) if b and float(f) > 1.0 and s == 0)
    fp_count_attack0 = sum(1 for b, f, s in zip(label_type, response_time, attack_sentence_num) if not b and float(f) < 1.0 and s == 0)
    tn_count_attack0 = sum(1 for b, f, s in zip(label_type, response_time, attack_sentence_num) if not b and float(f) > 1.0 and s == 0)

    # 135 total True and attack sentence 0
    # 135 total False and attack sentence 0
    tpr_attack0 = tp_count_attack0 / 135
    fpr_attack0 = fp_count_attack0 / 135
    fnr_attack0 = fn_count_attack0 / 135
    tnr_attack0 = tn_count_attack0 / 135

    print("tpr attack 0:", tpr_attack0)
    print("fnr 0:", fnr_attack0)
    print("tnr 0:", tnr_attack0)
    print("fpr 0:", fpr_attack0)

    cm_all(tpr_attack0, fnr_attack0, fpr_attack0, tnr_attack0)

    #cm_all(tpr, fnr, fpr, tnr)

    # get tpr fpr if you have two attacks (at least one of the two is successful):
    # get tpr
    same_answer = df['Same Answer'].to_list()
    response_time = df['Query Response Time'].to_list()
    label_type = df['Label Type'].to_list()
    attack_sentence_num = df['Attack Sentence Number'].to_list()

    # the 1 will come right after it if there's another attack sentence
    # should be over all instances where True label and 2 attack sentences
    #collapsed_label_type = []
    """ count = 0
    SKIP_NEXT_ONE = True
    total_2sentences = 0
    for idx, (b, f, s) in enumerate(zip(label_type, response_time, attack_sentence_num)):
        if b and ((s == 0 and idx+1 < len(label_type) and attack_sentence_num[idx+1] == 1) or s == 1):
            print(b, f, s)
            if s == 0:
                total_2sentences += 1
            # if s == 0 and response time < 1.0, add to arrays and skip if a 1 comes right after
            if s == 0 and float(f) < 1.0:
                SKIP_NEXT_ONE = True
                count += 1
            elif s == 0 and float(f) > 1.0:
                SKIP_NEXT_ONE = False # see if the next attack sentence worked
            elif not SKIP_NEXT_ONE and s == 1 and float(f) < 1.0:
                count += 1
            elif not SKIP_NEXT_ONE and s == 1 and float(f) > 1.0:
                continue
    
    print('tpr count 2 attack sentences:', count) # 78
    print('count 2 attack sentences:', total_2sentences) # 92 """
    # ~0.84

    

    """
    tpr: 0.5672514619883041
    fnr: 0.4327485380116959
    tnr: 0.9887640449438202
    fpr: 0.011235955056179775

    # the most similar attack queries!!!
    tpr attack 0: 0.837037037037037
    fnr 0: 0.16296296296296298
    tnr 0: 0.9777777777777777
    fpr 0: 0.022222222222222223
    """


    """
    Schedule a meeting for the interdisciplinary team to review and discuss treatment options for Ana with Poisoning
    Design an agenda for the interdisciplinary team meeting to discuss and optimize the treatment plan for Ana with Poisoning
    Design an agenda for the interdisciplinary team meeting to integrate patient preferences into the treatment plan for Ana with Poisoning
    """

    # the above is overall tpr, fpr, fnr, tnr
    # but we can also see it for aggregated by attack sentence number

    # true label type, true same answer/<0.6 second is true positive
    # true label type, false same answer/>0.6 second is false negative (should've been a cache hit)
    # false label type, false same answer/>0.6 is true negative
    # false label type, true same answer/<0.6 is false positive (ended up being semantically similar)

    # also we could just go by timing. < 0.6 seconds. this is more realistic actually, so let's do this

    """ print(same_answer[:10])
    print(response_time[:10])

    same_answer, response_time = sample_part(same_answer, response_time, 200)

    print(len(same_answer))
    print(len(response_time))

    #create_plot(same_answer, response_time)
    
    #create_plot_2(same_answer, response_time)
    print(same_answer.count(True))
    print(same_answer.count(False))

    same_answer, response_time = pick_50(same_answer, response_time)
    print(same_answer.count(True))
    print(same_answer.count(False)) """

    #create_plot(same_answer, response_time)



    return

def get_roc():
    df = pd.read_csv("./CacheHitMiss.csv")
    X = df[["Query Response Time"]]
    y = df["Same Answer"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=44)

    clf = LogisticRegression(penalty='l2', C=0.1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy", metrics.accuracy_score(y_test, y_pred))

    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label="Random Classifier")
    plt.legend(loc=4)
    
    plt.savefig("semantic_cache_roc.png", dpi=300)



    return

def get_roc2(): # 1 sentnece
    df = pd.read_csv("./CacheHitMiss.csv")
    df = df[df["Attack Sentence Number"] == 0]
    print(df.head)
    X = df[["Query Response Time"]]
    y = df["Label Type"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=44)

    clf = LogisticRegression(penalty='l2', C=0.1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy", metrics.accuracy_score(y_test, y_pred))

    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="Logistic Regression (C=0.1), AUC="+str(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label="Random Classifier")
    plt.legend(loc=4)
    
    plt.savefig("semantic_cache_label_type_roc.png", dpi=300)


    return

def get_roc2sentences():
    df = pd.read_csv("./CacheHitMiss.csv")
    df = df[df["Attack Sentence Number"] < 4]


    print(df.head)
    X = df[["Query Response Time"]]
    y = df["Label Type"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=44)

    clf = LogisticRegression(penalty='l2', C=0.1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy", metrics.accuracy_score(y_test, y_pred))

    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="Logistic Regression (C=0.1), AUC="+str(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label="Random Classifier")
    plt.legend(loc=4)
    
    plt.savefig("semantic_cache_label_type_roc_4attackSentences.png", dpi=300)


    return

def more_data():
    df = pd.read_csv('./CacheHitMiss.csv')
    hit_df = df[df['Same Answer'] == True] # all the cache hits

    average_cacheHit = hit_df['Query Response Time'].mean()
    std_deviation_cacheHit = hit_df['Query Response Time'].std()
    min_cacheHit = hit_df['Query Response Time'].min()
    max_cacheHit = hit_df['Query Response Time'].max()

    miss_df = df[df['Same Answer'] == False]
    #miss_df = miss_df[miss_df['Query Response Time'] != 0.166034460067749]
    #miss_df = miss_df[miss_df['Query Response Time'] != 0.1948654651641845]
    average_cacheMiss = miss_df['Query Response Time'].mean()
    std_deviation_cacheMiss = miss_df['Query Response Time'].std()
    min_cacheMiss = miss_df['Query Response Time'].min()
    max_cacheMiss = miss_df['Query Response Time'].max()

    miss_df_list = miss_df['Query Response Time'].to_list()

    for num in miss_df_list:
        if float(num) < .5:
            print(num)

    """
    0.2124781608581543
    0.2065081596374511
    0.2180299758911132
    0.166034460067749
    0.1986372470855713
    0.2069861888885498
    0.1948654651641845
    0.1972963809967041
    8 outliers for cache misses ... let's just report the avg 
    also this is out of 700 samples, so they really are outliers.
    """

    print('cache hit:', average_cacheHit, std_deviation_cacheHit, min_cacheHit, max_cacheHit)
    print('cache miss:', average_cacheMiss, std_deviation_cacheMiss, min_cacheMiss, max_cacheMiss)

    return

def main():
    #get_roc()
    #get_roc2()
    #more_data()
    get_roc2sentences()

    return

if __name__ == "__main__":
    main()