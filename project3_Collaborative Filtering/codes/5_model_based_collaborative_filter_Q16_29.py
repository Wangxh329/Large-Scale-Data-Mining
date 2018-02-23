from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.model_selection.validation import cross_validate
from surprise.model_selection.split import train_test_split
from surprise.dataset import Dataset
from surprise.reader import Reader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import csv

def main():
    # Load data
    reader = Reader(sep=',', rating_scale=(0.0, 5.0), skip_lines=1)
    allMoives = Dataset.load_from_file('ratings.csv', reader=reader)
    popMoives = Dataset.load_from_file('popular.csv', reader=reader)
    unpopMoives = Dataset.load_from_file('unpopular.csv', reader=reader)
    varMoives = Dataset.load_from_file('variance.csv', reader=reader)
    binary = []
    binary.append(Dataset.load_from_file('bin2.5.csv', reader=reader))
    binary.append(Dataset.load_from_file('bin3.csv', reader=reader))
    binary.append(Dataset.load_from_file('bin3.5.csv', reader=reader))
    binary.append(Dataset.load_from_file('bin4.csv', reader=reader))
    with open('movies.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader, None)
        movies = {int(movie[0]): movie[2] for movie in reader}

    # NMFs
    ks = range(2, 52, 2)
    mae, rmse = [0] * len(ks), [0] * len(ks)

    def nmf(dataName, data, biased=True):
        print('Start building NMF with ' + dataName + '!')
        for i, k in enumerate(ks):
            nmf = NMF(n_factors=k, biased=biased)
            scores = cross_validate(nmf, data, cv=10)
            mae[i] = scores['test_mae'].mean()
            rmse[i] = scores['test_rmse'].mean()
            print('k = '+str(k)+' finished!')
        plt.figure()
        plt.subplot(211)
        plt.plot(ks, mae)
        plt.xlabel('k')
        plt.ylabel('mean absolute error')
        plt.title('Mean absolute error vs. k of ' + dataName)
        plt.subplot(212)
        plt.plot(ks, rmse)
        plt.xlabel('k')
        plt.ylabel('root mean squared error')
        plt.title('Root mean squared error vs. k of ' + dataName)
        print('mae:')
        print(mae)
        print('rmse:')
        print(rmse)
        print('Finish building NMF with ' + dataName + '!')

    # Q17
    nmf('all movies', allMoives)

    # Q18
    optimalK = 4
    print('The optimal number of latent factors is ' + str(optimalK))
    
    # Q19
    nmf('popular movies', popMoives)
    
    # Q20
    nmf('unpopular movies', unpopMoives)
    
    # Q21
    nmf('high variance movies', varMoives)

    # Draw ROC Curve
    thresholds = [2.5, 3, 3.5, 4]

    def drawRoc(model, i, k):
        print('Start drawing ROC curve of NMF with optimal k = ' + str(k) + ', threshold = ' +
              str(thresholds[i]) + '!')
        train, test = train_test_split(binary[i], train_size=0.9, test_size=0.1)
        model.fit(train)
        labels = model.test(test)
        y_true = [label.r_ui for label in labels]
        y_pred = [label.est for label in labels]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of NMF with optimal k = ' + str(k) + ', threshold = ' + str(thresholds[i]))
        plt.legend(loc="lower right")
        print('Finish drawing ROC curve of NMF with optimal k = ' + str(k) + ', threshold = ' +
              str(thresholds[i]) + '!')

    # Q22
    nmf = NMF(n_factors=optimalK)
    for i in range(len(thresholds)):
        drawRoc(nmf, i, optimalK)

    # Q23
    print("Start finding top K!")
    k, col = 20, 5
    nmf = NMF(n_factors=k)
    trainAllMovies = allMoives.build_full_trainset()
    nmf.fit(trainAllMovies)
    ids = [[] for _ in range(col)]
    for i in range(col):
        factors = nmf.qi[:, i]
        s = sorted([[i, factor] for i, factor in enumerate(factors)], key=lambda x:x[1], reverse=True)
        for k in range(10):
            ids[i].append(s[k][0])
    genres = [[] for _ in range(col)]
    for i in range(col):
        for j in range(10):
            genres[i].append(movies[int(trainAllMovies.to_raw_iid(ids[i][j]))])
    for i in range(col):
        print('Col ' + str(i+1) + ':')
        for genre in genres[i]:
            print(genre, end=', ')
        print('')
    print("Finish finding top K!")

    # Q24
    nmf('all movies', allMoives, True)
    
    # Q25
    optimalKBiased = 2
    print('The optimal number of latent factors is ' + optimalKBiased)
    
    # Q26
    nmf('popular movies', popMoives, True)
    
    # Q27
    nmf('unpopular movies', unpopMoives, True)
    
    # Q28
    nmf('high variance movies', varMoives, True)

    # Q29
    optimalKBiased = 2
    nmfBiased = NMF(n_factors=optimalKBiased, biased=True)
    for i in range(len(thresholds)):
        drawRoc(nmfBiased, i, optimalKBiased)

    plt.show()

if __name__ == '__main__':
    main()