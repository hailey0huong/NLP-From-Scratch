'''
Implement Naive Bayes model to classify sentiment

'''

import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.vocab_len = data.X.shape[1]
        self.count_positive = 0
        self.count_negative = 0
        self.num_positive_reviews = 0 
        self.num_negative_reviews = 0
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0
        self.P_negative = 0
        self.deno_pos = 0
        self.deno_neg = 0
        self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        
        #Get number of positive reviews?
        self.num_positive_reviews = len(positive_indices)
        self.num_negative_reviews = len(negative_indices)
        
        #Sum by columns for positive and negative docs for count_positive and count_negative
        self.count_positive = np.sum(X[positive_indices,:], axis = 0)
        self.count_negative = np.sum(X[negative_indices,:], axis = 0)

        self.total_positive_words = np.sum(self.count_positive, axis = 1).item()
        self.total_negative_words = np.sum(self.count_negative, axis = 1).item()
        

        self.deno_pos = self.total_positive_words + self.ALPHA*self.vocab_len
        self.deno_neg = self.total_negative_words + self.ALPHA*self.vocab_len


        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, test, probThresh = None):
        #TODO: Implement Naive Bayes Classification
        self.P_positive = self.total_positive_words/(self.total_negative_words+self.total_positive_words)
        self.P_negative = self.total_negative_words/(self.total_negative_words+self.total_positive_words)

        if not probThresh:
            pred_labels = []
            X = test.X
            
            sh = X.shape[0]

            for i in range(sh):
                z = X[i].nonzero() #get all indexes that have values != 0
                score_pos  = log(self.P_positive)
                score_neg = log(self.P_negative)
                for j in range(len(z[1])):
                    # Look at each feature
                    col_index = z[1][j]
                    if col_index >= self.count_positive.shape[1]:
                        score_pos += X[i,col_index]*log((0.0 + self.ALPHA)/self.deno_pos)
                        score_neg += X[i,col_index]*log((0.0 + self.ALPHA)/self.deno_neg)
                    else:
                        score_pos += X[i,col_index]*log((self.count_positive[0,col_index]+self.ALPHA)/ self.deno_pos)
                        score_neg += X[i,col_index]*log((self.count_negative[0,col_index]+self.ALPHA)/ self.deno_neg)
    
                if score_pos > score_neg:            # Predict positive
                    pred_labels.append(1.0)
                else:               # Predict negative
                    pred_labels.append(-1.0)    
        
        else: #Use predictProb
            indexes = np.arange(test.X.shape[0])
            pred_labels = self.PredictProb(test, indexes, probThresh=probThresh)

        return np.array(pred_labels)

    def LogSum(self, logx, logy):   
        # TO DO: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m)) #log(x+y)

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test, indexes, probThresh = 0.5):
        
        pred_labels = []
        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            predicted_label = 0
            z = test.X[i].nonzero()

            log_prob_pos = log(self.P_positive)
            log_prob_neg = log(self.P_negative)

            for j in range(len(z[1])):
                row_index = i
                col_index = z[1][j]
                if col_index >= self.vocab_len:
                    log_prob_pos += test.X[i,col_index]*log((0.0 + self.ALPHA)/ self.deno_pos)
                    log_prob_neg += test.X[i,col_index]*log((0.0 + self.ALPHA)/ self.deno_neg)
                else:
                    log_prob_pos += test.X[i,col_index]*log((self.count_positive[0,col_index] + self.ALPHA)/ self.deno_pos)
                    log_prob_neg += test.X[i,col_index]*log((self.count_negative[0,col_index] + self.ALPHA) /self.deno_neg)
            
            predicted_prob_positive = exp(log_prob_pos - self.LogSum(log_prob_pos,log_prob_neg))
            predicted_prob_negative = exp(log_prob_neg - self.LogSum(log_prob_pos,log_prob_neg))

            if predicted_prob_positive > probThresh:
                predicted_label = 1.0
            else:
                predicted_label = -1.0
            
            pred_labels.append(predicted_label)
            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            #print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])
        return pred_labels

    # Evaluate performance on test data 
    def Eval(self, test):
        Y_pred = self.PredictLabel(test)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()

    def EvalPrecision(self, Y_true, Y_pred, label = 'positive'):
        if label == 'positive': #positve class
            TP = 0
            for i in range(Y_pred.shape[0]):
                if (Y_pred[i] == 1.0) & (Y_true[i] == 1.0):
                    TP +=1
            count_preds_pos = sum(Y_pred == 1.0 )
            #In case there is no positive prediction
            try: 
                return TP/count_preds_pos
            except:
                print ('division by zero')
                return 0
        else: #negative class
            TN = 0
            for i in range(Y_pred.shape[0]):
                if (Y_pred[i] == -1.0) &(Y_true[i] == -1.0):
                    TN+=1
            count_preds_neg = sum(Y_pred == -1.0)
            #In case no negative prediction
            try: 
                return TN/count_preds_neg
            except:
                print ('division by zero')
                return 0

    
    def EvalRecall(self, Y_true, Y_pred, label = 'positive'):
    
        if label == 'positive': #positve class
            TP = 0
            for i in range(Y_pred.shape[0]):
                if (Y_pred[i] == 1.0) & (Y_true[i] == 1.0):
                    TP +=1
            count_true_pos = sum(Y_true == 1.0 )
            #In case there is no true positive
            try: 
                return TP/count_true_pos
            except:
                print ('division by zero')
                return 0
        else:  #negative class 
            TN = 0
            for i in range(Y_pred.shape[0]):
                if (Y_pred[i] == -1.0) &(Y_true[i] == -1.0):
                    TN+=1
            count_true_neg = sum(Y_true == -1.0)
            #In case no true negative
            try: 
                return TN/count_true_neg
            except:
                print ('division by zero')
                return 0
    
    def precision_recall_curve(self, test, label = 'positive'):
        #Get data
        thresholds =  np.arange(0.1,1,.1)
        precision = []
        recall = []
        for i in thresholds:
            Y_pred = self.PredictLabel(test, probThresh=i)
            precision_score = self.EvalPrecision(Y_true = test.Y, Y_pred = Y_pred, label = label )
            recall_score = self.EvalRecall(Y_true = test.Y, Y_pred = Y_pred, label = label)
            precision.append(precision_score)
            recall.append(recall_score)

        #Plot the curve
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure()
    
        plt.plot(recall, precision, marker = '.')
        #plt.fill_between(recall, precision, alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - {} class'.format(label))
        plt.xlim([0.5,1.0])
        plt.ylim([0.5,1.0])
        fig.savefig('precision_recall_curve_{}.png'.format(label))

    def find_top_words(self, top_n = 20):

        #Compute the weights for each word
        weights_pos_words = np.array(self.count_positive/ self.total_positive_words)
        weights_neg_words = np.array(self.count_negative/self.total_negative_words)

        #Get top wordID:
        top_list_pos = weights_pos_words.argsort()[0][::-1][:top_n]
        top_list_neg = weights_neg_words.argsort()[0][::-1][:top_n]

        
        #Get top weights:
        top_weights_pos = weights_pos_words[0,top_list_pos]
        top_weights_neg = weights_neg_words[0,top_list_neg]

        #Convert wordID to word
        top_words_pos = [self.data.vocab.GetWord(i) for i in top_list_pos]
        top_words_neg = [self.data.vocab.GetWord(i) for i in top_list_neg]

        #Print results
        print('Top Positive Words')
        for word, weight in zip(top_words_pos, top_weights_pos):
            print(word + ' '+str(round(weight,5)), end = ' ; ')
        print('\n')
        print('Top Negative Words')
        for word, weight in zip(top_words_neg, top_weights_neg):
            print(word+' '+str(round(weight,5)), end = ' ; ')
        print('\n')
        





if __name__ == "__main__":
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    traindata.vocab.locked = False
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)   
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))
    nb.PredictProb(testdata, np.arange(10)) #Print 10 reviews
    #print("Plot precision and recall curve")
    #nb.precision_recall_curve(testdata, label = sys.argv[3])
    #nb.find_top_words(top_n = 20)


