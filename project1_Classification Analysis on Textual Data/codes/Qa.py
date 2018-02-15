
# coding: utf-8

# In[13]:


from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt

# define two classes
computer_technology_class = [['comp.graphics'], ['comp.os.ms-windows.misc'], ['comp.sys.ibm.pc.hardware'], ['comp.sys.mac.hardware']]
recreational_activity_class = [['rec.autos'], ['rec.motorcycles'], ['rec.sport.baseball'], ['rec.sport.hockey']]
subclasses_of_ducuments = computer_technology_class + recreational_activity_class

# define a function to count number of documents and plot a histogram
def plot_histogram(target_set):
    substring_pair = {'all': '(all subsets):', 'train': '(train subsets)', 'test': '(test subsets)'}
    number_of_documents = []
    number_of_documents_in_comp_tech = 0
    number_of_documents_in_rec_act = 0
    # load data of computer technology subclass from datasets
    for i in range(4):
        number = len(fetch_20newsgroups(subset=target_set, categories=computer_technology_class[i], shuffle=True, random_state=42).data)
        number_of_documents.append(number)
        number_of_documents_in_comp_tech += number 
    # load data of recreational activity subclass from datasets
    for j in range(4):
        number = len(fetch_20newsgroups(subset=target_set, categories=recreational_activity_class[j], shuffle=True, random_state=42).data)
        number_of_documents.append(number)
        number_of_documents_in_rec_act += number
    # plot histogram
    x_labels = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    fig, ax = plt.subplots()
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'pink']
    ax.set_xticks([i+0.25 for i in range(1,9)])
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize = 12)
    
    rects = plt.bar([i for i in range(1,9)], number_of_documents, 0.5, align='edge', alpha = 0.8, color = color)
    plt.xlabel('Topics', fontsize = 14)
    plt.ylabel('Number of Documents', fontsize = 14)
    plt.title('Number of documents per topic ' + substring_pair[target_set], fontsize = 18)
    plt.axis([0.5,9,0,1100])
    
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height, '%d' % int(height), ha='center', va='bottom')
    
    plt.show()
    # count number of documents in each class
    print 'Number of documents in Computer Technology: ' + str(number_of_documents_in_comp_tech)
    print 'Number of documents in Recreational Activity: ' + str(number_of_documents_in_rec_act)

# plot histograms of all subsets, train subsets and test subsets
print '1. all subsets'
plot_histogram('all')
print ' '
print '2. train subsets'
plot_histogram('train')
print ' '
print '3. test subsets'
plot_histogram('test')

