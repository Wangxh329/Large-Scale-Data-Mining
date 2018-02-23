
# coding: utf-8

# In[6]:


from collections import defaultdict
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.model_selection import KFold
from surprise import Dataset
from surprise import Reader
import os
import matplotlib.pyplot as plt


def precision_recall_at_t(predictions, t, threshold=3):
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Number of recommended items
        n_rec_k = t
        
        # Number of relevant and recommended items in top t
        n_rel_and_rec_k = 0
        for i in range(t):
            if user_ratings[i][1] >= threshold:
                n_rel_and_rec_k += 1
        
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel

    return precisions, recalls

# If |G| = 0 for some user in the test set, then drop this user
# If some user in the test set has rated less than t items, then drop this user
def testset_trim(testset, t, threshold=3):
    count = {}
    likes = {}
    
    for (u, i, r) in testset:
        if u not in count:
            count[u] = 0
        count[u] += 1
        if u not in likes:
            likes[u] = 0
        if r >= threshold:
            likes[u] += 1
            
    res = []
    for (u, i, r) in testset:
        if count[u] >= t and likes[u] > 0:
            res.append((u, i, r))
    
    return res


def draw_t_prec_recall(algo, kf, t_low, t_high, thre):
    kf = KFold(n_splits=kf)
    ts = [i for i in range(t_low, t_high + 1)]
    precision = []
    recall = []
    
    for t in ts:
        temp_prec = []
        temp_recall = []
        for trainset, testset in kf.split(data):

            # train and test algorithm.
            algo.fit(trainset)

            trimmed_testset = testset_trim(testset, t, threshold=thre)
            predictions = algo.test(trimmed_testset)
            precisions, recalls = precision_recall_at_t(predictions, t, threshold=thre)

            fold_mean_prec = sum(prec for prec in precisions.values()) / len(precisions)
            fold_mean_recall = sum(rec for rec in recalls.values()) / len(recalls)
            
            temp_prec.append(fold_mean_prec)
            temp_recall.append(fold_mean_recall)

        t_mean_prec = sum(prec for prec in temp_prec) / len(temp_prec)
        t_mean_recall = sum(rec for rec in temp_recall) / len(temp_recall)
        precision.append(t_mean_prec)
        recall.append(t_mean_recall)
    return ts, precision, recall

# read in data
file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating', sep=',',skip_lines=1, rating_scale=(0.5, 5))
data = Dataset.load_from_file(file_path, reader=reader)

sim_options = {'name': 'pearson'}
knn = KNNWithMeans(k=24, sim_options=sim_options)
nmf = NMF(n_factors=4)
nmfBiased = NMF(n_factors=2, biased=True)

algs = []
algs.append(knn)
algs.append(nmf)
algs.append(nmfBiased)

names = {}
names[knn] = "KNN"
names[nmf] = "NNMF"
names[nmfBiased] = "NMF(biased)"

res_t_p_r = {}
for alg in algs:
    print (names[alg])
    ts, precision, recall = draw_t_prec_recall(alg, 10, 1, 25, 3)
    res_t_p_r[names[alg]] = (ts, precision, recall)

# draw precision-recall curve
def draw_curve(model):
    (ts, precision, recall) = res_t_p_r[model]
    plt.figure(figsize=(8,6))
    plt.scatter(ts, precision, color='deepskyblue', lw=2)
    plt.xlabel('Recommendation List Size (t)', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('Precision plot with ' + model, fontsize=23)
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.scatter(ts, recall, color='deepskyblue', lw=2)
    plt.xlabel('Recommendation List Size (t)', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.title('Recall plot with ' + model, fontsize=23)
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.scatter(recall, precision, color='deepskyblue', lw=2)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('Precision vs Recall with ' + model, fontsize=23)
    plt.show()

# Q36
print("KNN filter")
draw_curve(names[knn])
print("")

# Q37
print("NNMF filter")
draw_curve(names[nmf])
print("")

# Q38
print("MF with bias filter")
draw_curve(names[nmfBiased])
print("")

# Q39
(knn_ts, knn_precision, knn_recall) = res_t_p_r[names[knn]]
(nnmf_ts, nnmf_precision, nnmf_recall) = res_t_p_r[names[nmf]]
(mf_bias_ts, mf_bias_precision, mf_bias_recall) = res_t_p_r[names[nmfBiased]]
plt.figure(figsize=(12,9))
plt.scatter(knn_recall, knn_precision, color='deepskyblue', lw=2, label='kNN collaborative filter')
plt.scatter(nnmf_recall, nnmf_precision, color='deeppink', lw=2, label='NNMF collaborative filter')
plt.scatter(mf_bias_recall, mf_bias_precision, color='darkorange', lw=2, label='MF with bias collaborative filter')
plt.xlabel('Recall', fontsize=16)
plt.ylabel('Precision', fontsize=16)
plt.title('Precision vs Recall Comparison', fontsize=23)
plt.legend(loc="lower left")
plt.show()

