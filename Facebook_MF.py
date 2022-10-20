#try:
import numpy
import math
import pandas
import cPickle as pickle
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
#from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from math import sqrt
import random
import scipy.sparse as sparse

import time
#from ggplot import *
#import SemEMF
from numba import typeof, double, int_
from numba.decorators import autojit, jit

#except:
#    print "This implementation requires the numpy module."
#    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""

@autojit(locals={'step': int_, 'e': double, 'err': double})
def matrix_factorization(R, P, Q, K, E, steps, alpha, beta):
    Q = Q.T
    for step in range(steps):

        print('step ',step)
        for i in range(len(R)):
            #print '1'
            for j in range(len(R[i])):
                #print '2'
                if R[i][j] > 0:
                    #print i
                    #print j
                    #print R[i][j]
                    #print '3'
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    #print '4'
                    for k in range(K):
                        #print '5'
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        count = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    count = count + 1
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    #e1 = e
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        print step
        print count
#        print ('RMSE ', math.sqrt(e / count))
        print ('error', math.sqrt(e/count))
        if e < 0.001:
            break

    pickle.dump(P, open("../Experiments/Facebook_MF/P/"+str(E)+"_P60k_k"+str(K)+"_alpha"+str(alpha)+"_beta"+str(beta)+"_first_90train", "wb"))
    pickle.dump(Q, open("../Experiments/Facebook_MF/Q/"+str(E)+"_Q60k_k"+str(K)+"_alpha"+str(alpha)+"_beta"+str(beta)+"_first_90train", "wb"))
    #pickle.dump(P, open("../Experiments/MF_test25/P/" + str(E) + "_P60k_k" + str(K) + "_alpha" + str(alpha) + "_beta" + str(beta) + "_first_90train", "wb"))
    #pickle.dump(Q, open("../Experiments/MF_test25/Q/" + str(E) + "_Q60k_k" + str(K) + "_alpha" + str(alpha) + "_beta" + str(beta) + "_first_90train", "wb"))

    return P, Q

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations):
    # Get and sort the user's predictions
    user_row_number = userID - 1  # UserID starts at 1, not 0
    #predictions_df1 = predictions_df.as_matrix()
    #sorted_user_predictions1 = numpy.argsort(-predictions_df1[user_row_number])
    #sorted_user_predictions = pandas.DataFrame(sorted_user_predictions1, columns=R_norm_train1[0].columns, dtype=float)
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how='left', left_on='MovieID', right_on='MovieID').sort_values(['Rating'], ascending=False))

    #    print '_________user_full'
    #    print user_full

    print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print ('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
                           merge(pandas.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='MovieID',
                                 right_on='MovieID')
                           .
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1]
                           )

    return user_full, recommendations

def rmse(predictions, actual):

    count = 0
    differences_squared = 0
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            #print len(actual[i])
            if actual[i][j] > 0:
                differences = predictions[i][j] - actual[i][j]  # the DIFFERENCEs.
    #differences = predictions - actual
    #print predictions[424]
    #print actual[424]
    #print differences[424]
                differences_squared = differences_squared + differences ** 2       #the SQUAREs of ^
                count = count + 1

                #sumds = sumds + differences_squared
    #mean_of_differences_squared = differences_squared/float(count)  #the MEAN of ^
    print count
    mean_of_differences_squared = differences_squared/count  #the MEAN of ^
    #mean_of_differences_squared = differences_squared / len(actual[0])  # the MEAN of ^
    #mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    #mean_of_differences_squared = sumds.mean()  # the MEAN of ^
    rmse_val = numpy.sqrt(mean_of_differences_squared)           #ROOT of ^
    #print ('Data Size: ', count)
    return rmse_val

def mae(predictions, actual):
    differences_sum = 0
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            if actual[i][j] > 0:
                differences_sum = numpy.sum(numpy.absolute(predictions[i][j] - actual[i][j]))  # the sum of absolute DIFFERENCEs.
    return differences_sum

def getRMSE(Actual_Rating, Predicted_Rating):
    # Calculate the Root Mean Squared Error(RMS)
    rmse = 0.0
    for i in range(len(Actual_Rating)):
        for j in range(len(Actual_Rating[0])):
            if Actual_Rating[i][j] > 0:
                rmse = rmse + pow((Actual_Rating[i][j] - Predicted_Rating[i][j]), 2)

    rmse = rmse * 1.0 / len(Actual_Rating)
    print len(Actual_Rating)
    #rmse = rmse.mean()
    rmse = math.sqrt(rmse)

    # Print and return the RMSE
    print 'Root Mean Squared Error(RMS) = ' , rmse
    return rmse


def topn(R, n, u, T):  # Calculate the list of n recommended items
    sorted = numpy.argsort(R[u])[::-1]
    #R1 = [i for i in sorted if i not in T[u]]
    top_n = sorted[0:n]
    return top_n

    # function for MEP
def calculate_MEP(eR,expl,test_user,T,n): # Calculate MEP
    MEP = 0
    EP = 0
    for u in range(len(test_user)):
        top_n = topn(eR, n, u, T)
        EP = len([j for j in range(len(eR[0])) if expl[u,j] > 0 and j in top_n]) / float(len(top_n))
        MEP = MEP + EP
    MEP = MEP/float(len(test_user))
    return MEP
    #Function for MER
def calculate_MER(eR,expl,test_user,T,n): # calculate MER
    MER = 0
    ER = 0
    for u in range(len(test_user)):
        top_n = topn(eR, n, u, T)
        ER = len([j for j in range(len(eR[0])) if expl[u,j] > 0 and (j in top_n)]) / float(len([j for j in range(len(eR[0])) if expl[u,j] > 0 ]))
        MER = MER + ER
    MER = MER/float(len(test_user))
    return MER

def calculate_MAP(actual, predicted, n):
    MAP = 0
    AP = 0
    c = 0
    for u in range(len(actual)):
        top_n = topn(predicted, n, u)
        #print top_n
        for j in range(len(actual[0])):
            if actual[u][j] > 0 and j in top_n:
                c = c + 1
                #print actual[u][j]
                AP = c / float(len(top_n))
    print(AP)
    MAP = MAP + AP
    MAP = MAP / float(len(actual))
    return MAP

def dcg_score(y_true, y_score, n, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = numpy.argsort(y_score)[::-1]
    y_true = numpy.take(y_true, order[:n])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = numpy.log2(numpy.arange(len(y_true)) + 2)
    return numpy.sum(gains / discounts)


def ndcg_score1(y_true, y_score, n, gains="linear"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, n, gains)
    actual = dcg_score(y_true, y_score, n, gains)
    return actual / best

def calc_exp(rate,neighbor):    #calculate the explainability table
    expl = numpy.zeros(rate.shape)
    dist_items = metrics.pairwise.pairwise_distances(rate, Y=None, metric='cosine', n_jobs=-1)
    sorted_ind = numpy.argsort(dist_items, axis=1) #increasing order based on distance therefore index 0 has the most similar(itself)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            #if rate[i][j] >0 :
            sims = sorted_ind[i,1:(neighbor+1)]  #index of the neighbors
            temp = rate[sims,j]
            num = len(numpy.where(temp>0)[0])/ float(neighbor)
            expl[i][j] = num
    return expl

###############################################################################

# Following function divides the data into train-test randomly

"""
def spilt_rating_dat(data, size=0.2):
    train_data = []
    test_data = []
    for line in data:
        rand = random.random()
        if rand < size:
            test_data.append(line)
        else:
            train_data.append(line)
    train_data = array(train_data)
    test_data = array(test_data)
    return train_data, test_data
"""

def calc_exp(rate,neighbor):    #calculate the explainability table
    expl = numpy.zeros(rate.shape)
    dist_items = metrics.pairwise.pairwise_distances(rate, Y=None, metric='cosine', n_jobs=-1)
    sorted_ind = numpy.argsort(dist_items, axis=1) #increasing order based on distance therefore index 0 has the most similar(itself)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            #if rate[i][j] >0 :
            sims = sorted_ind[i,1:(neighbor+1)]  #index of the neighbors
            temp = rate[sims,j]
            num = len(numpy.where(temp>0)[0])/ float(neighbor)
            expl[i][j] = num
    return expl

def topnR(R, n, test_R, u):
    # print(self.test_R[u])
    #numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)
    sorting = numpy.argsort(R[u])[::-1]
    #print('R_Sorted_For_ALL',sorting)
    R_in_Test = []
    R_in_Test1 = []
    for i in sorting:
        if test_R[u][i] > 0:    # threshold
            R_in_Test.append(i)

    c = (10-len(R_in_Test))
    d = str(10-len(R_in_Test))
    #print('R_Sorted_Only_in_Test: ',R_in_Test)
    if len(R_in_Test)<10:

        for j in sorting:
            if j not in R_in_Test:
                if c != 0:
                    R_in_Test.append(j)
                    c = c - 1
    #if len(R_in_Test)<10:
        #a = sorting[0:(10 - len(R_in_Test))]

        #R_in_Test1.append(sorting[0:10-len(R_in_Test)].tolist())
    #print('Length of u: ',u,len(R_in_Test))
    #print(R_in_Test)
    #R_in_Test_final = R_in_Test+R_in_Test1
    #R_in_Test_final = numpy.column_stack((R_in_Test, R_in_Test1))
    #R_in_Test_final = []


    #print('R_Sorted_Only_in_Test_Plus_missing10: ', R_in_Test1)
    #print('R_Sorted_Only_in_Test_Plus_missing10_2: ', R_in_Test2)

    #print('R_Sorted_Only_in_Test_Plus_missing10_correct: ', R_in_Test_final)

    #sorting = numpy.argsort(R[u])[::-1]
    #print('topn1: ', u, 'Sorting',sorting)
    #print(sorting)

    top_nR = R_in_Test[0:n]
    #print('Prediction Sorted: ',top_n)
    #top_n = sorting[0:n]
    #print('Prediction: ', top_n)
    return top_nR



def topnT(R, n, test_R, u):
    match_test_Est = []
    #numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)

    sorting = numpy.argsort(test_R[u])[::-1]

    #print('Length: ',len(sorting))
    for i in sorting:
        if test_R[u][i] > 0:
            match_test_Est.append(i)
    c = (10 - len(match_test_Est))
    d = str(10 - len(match_test_Est))
    # print('R_Sorted_Only_in_Test: ',R_in_Test)
    if len(match_test_Est) < 10:

        for j in sorting:
            if j not in match_test_Est:
                if c != 0:
                    match_test_Est.append(j)
                    c = c - 1
    #print('topn_1', u, 'EST :', match_test_Est)
    #top_n = sorting[0:n]
    #top_n = match_test_Est

    #top_nT = match_test_Est[0:n]
    top_nT = match_test_Est
    #print('Test Sorted: ', top_n)
    #print('______________________________________')
    #print('True: ', top_n)
    return top_nT

def nDCG(ltrue, lrec):


    DCG = 0
    if lrec[0] in ltrue:
        DCG = DCG + 1
    for i in range(1, len(lrec)):
        if lrec[i] in ltrue:
            DCG = DCG + 1 / (numpy.log2(i + 1))

    IDCG = 0

    k = 0

    for i in range(0, len(lrec)):
        if lrec[i] in ltrue:
            k = k + 1
            if k == 1:
                IDCG = 1
            else:
                IDCG = IDCG + 1 / (numpy.log2(k))
    if IDCG == 0:
        return 0
    else:
        return DCG / IDCG


def calculate_ndcg(R, test_R, n):
    M, N = R.shape
    return sum(nDCG(topnT(R, n, test_R, u), topnR(R, n, test_R, u)) for u in range(M)) / M
    #return sum(ndcg_score1(topn_l(R, n, test_R, u), topn1(R, n,test_R, u), n) for u in range(M)) / M
    #u = 750
    #return nDCG(topn_l(R, n, test_R, u), topn1(R, n, test_R, u))
    #return ndcg_score1(topn_l(R, n, test_R, u), topn1(R, n, test_R, u), n)


def topnRmap(R, n, u):
    sorting = numpy.argsort(R[u])[::-1]
    top_nR = sorting[0:n]
    return top_nR


def topnTmap(test_R, u):
    match_test_Est = []
    for i,k in enumerate(test_R[u]):
        if test_R[u][i] > 0:
            match_test_Est.append(i)
    top_nT = match_test_Est
    return top_nT


def ap_k(ord_pred, target, k):
    #hits = []
    if len(ord_pred) > k:
        ord_pred = ord_pred[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(ord_pred):
        if p in target:
            #hits.append(p)
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not target:
        return 1.0

    return score / min(len(target), k)#, hits


# Calculates the mean average precision at k
def map_k(ord_pred_all, target_all, k):
    return numpy.mean([ap_k(p, a, k) for p, a in zip(ord_pred_all, target_all)])


def auc(ord_pred, target):
    """
    @param ord_pred: ordere predicted scores
    @type ord_pred: iterable sequence of int
    @param target: set of positive items
    @type target: set of int
    @return: AUC
    @rtype: float
    """
    np = len(target)
    n = len(ord_pred)
    area = 0.0

    for i in range(n):
        if ord_pred[i] in target:
            for j in range(i + 1, n):
                if ord_pred[j] not in target:
                    area += 1.0

    if float(np * (n - np)) == 0:
        area = 0
    else:
        area /= float(np * (n - np))
    return area


def ndcg_k(ord_pred, target, k):
    k = min(k, len(ord_pred))
    idcg = idcg_k(k)
    dcg_k = sum([int(ord_pred[i] in target) / math.log(i + 2, 2) for i in xrange(k)])
    return dcg_k / idcg


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in xrange(k)])
    if not res:
        return 1.0
    else:
        return res



if __name__ == "__main__":

    train_list_original = [i.strip().split(",") for i in open('../Data/matrix_fatorization_data_indexed_3_normalized_Train.csv', 'r').readlines()]
    train_df_original = pandas.DataFrame(train_list_original, columns=['uid', 'fid', 'interactions'], dtype=float)
    R_norm_train1 = train_df_original.pivot(index='uid', columns='fid', values='interactions').fillna(0)
    R_norm_train = R_norm_train1.as_matrix()

    test_list_original = [i.strip().split(",") for i in open('../Data/matrix_fatorization_data_indexed_3_normalized_Test.csv', 'r').readlines()]
    test_df_original = pandas.DataFrame(test_list_original, columns=['uid', 'fid', 'interactions'], dtype=float)
    R_norm_test1 = test_df_original.pivot(index='uid', columns='fid', values='interactions').fillna(0)
    R_norm_test = R_norm_test1.as_matrix()

    '''
    #ratings_list_original = [i.strip().split(",") for i in open('../Data/matrix_fatorization_data_indexed_3_normalized.csv', 'r').readlines()]
    ratings_list_original = [i.strip().split(",") for i in open('../Data/rating_final1.csv', 'r').readlines()]

    #ratings_df_original = pandas.DataFrame(ratings_list_original, columns=['uid_index', 'uid', 'fid_index', 'fid', 'interactions'], dtype=int)
    ratings_df_original = pandas.DataFrame(ratings_list_original, columns=['uid_index', 'fid_index', 'interactions'], dtype=int)

    data = ratings_df_original.pivot(index='uid_index', columns='fid_index', values='interactions').fillna(0)
    data1 = data.as_matrix()
    '''

    data = pandas.read_csv("../Data/matrix_fatorization_data_indexed_3_normalized.csv", sep=',', names="uid,fid,interactions".split(","))
    train_data = pandas.read_csv("../Data/matrix_fatorization_data_indexed_3_normalized_Train.csv", sep=',', names="uid,fid,interactions".split(","))
    test_data = pandas.read_csv("../Data/matrix_fatorization_data_indexed_3_normalized_Test.csv", sep=',', names="uid,fid,interactions".split(","))

    #alldata = pandas.read_csv("../Restaurant Data with Consumer Ratings/Merged1.csv", sep=',', names="userID,uniqueUserID,placeID,uniquePlaceID,rating,food_rating,service_rating,cuisine,cuisineID".split(","))
    #cuisinedata = pandas.read_csv("../Restaurant Data with Consumer Ratings/chefmozcuisine.csv", sep=',', names="placeID,Rcuisine".split(","))
    
    n_users = data.uid.unique().shape[0]
    n_items = data.fid.unique().shape[0]
    print '11111111'
    print(n_users)
    print '22222222'
    print(n_items)
    #n_cuisine = cuisinedata.Rcuisine.unique().shape[0]



    #from sklearn import cross_validation as cv

    #train_data, test_data = cv.train_test_split(data, test_size=0.3)

    # Create two user-item matrices, one for training and another for testing
    train_data_matrix = numpy.zeros((n_users, n_items))
    print train_data_matrix.shape
    for line in train_data.itertuples():
        #print line
        #print line[1]
        #print line[2]
        #print line[3]
        train_data_matrix[int(line[1]) - 1, int(line[2]) - 1] = int(line[3])

    test_data_matrix = numpy.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[int(line[1]) - 1, int(line[2]) - 1] = int(line[3])

    #print train_data_matrix
    #print test_data_matrix

    '''
    ############################################################
    # Construct user-feature matrix
    ############################################################

    uF = numpy.zeros((n_users, n_cuisine))

    for i in range(len(alldata)):
        s = int(alldata['uniqueUserID'][i])
        s1 = int(alldata['cuisineID'][i])
        uF[s - 1][s1 - 1] = uF[s - 1][s1 - 1] + 1

    ############################################################
    # Construct item-feature matrix
    ############################################################

    iF = numpy.zeros((n_items, n_cuisine))

    for i in range(len(alldata)):
        d = int(alldata['uniquePlaceID'][i])
        d1 = int(alldata['cuisineID'][i])
        iF[d - 1][d1 - 1] = 1
    # print(iF[0])
    iF = iF[~numpy.all(iF == 0, axis=1)]
    
    ##################################################
    #  Construct the explainability Semantic graph W
    ##################################################

    SW = numpy.dot(uF, iF.T)
    SW_norm = numpy.zeros(SW.shape)
    # coo = 0
    for i in range(len(SW_norm)):
        # for i in range(0, 1, 1):
        # print(i)
        for j in range(len(SW_norm[0])):
            # for j in range(1,2,1):
            if SW[i, j] > 0:
                # coo = coo + 1
                num = (SW[i, j] - numpy.min(SW[i])) / float((numpy.max(SW[i]) - numpy.min(SW[i])))
                SW_norm[i, j] = num

'''
    #W = calc_exp(train_data_matrix, 50)
#    W = calc_exp(data, 50)

    eval = []
    t = True
    for K in range(1, 6, 1):

        rmse1_E = 0
        rmse2_E = 0
        mae1_E = 0
        mae2_E = 0
        MEP_E = 0
        MER_E = 0
        MAP_E = 0
        MAR_E = 0
        EF1_E = 0
        F1_E = 0
        ndcg_E = 0
        auc_E = 0

        for E in range(1, 11, 1):

            print('# of features ', K)
            print('# of experiment ', E)

            #n_users = data.uid_index.unique().shape[0]
            #n_items = data.fid_index.unique().shape[0]
            #print n_users
            #print n_items
            N = len(R_norm_train)
            M = len(R_norm_train[0])
            #N = len(data1)
            #print data.shape
            #print N
            #M = len(data1[0])
            #print M
            # K = 10
            P = numpy.random.rand(N, K)
            Q = numpy.random.rand(M, K)
            steps = 20
            alpha = 0.01
            beta = 0.001

            #nP, nQ = matrix_factorization(R_norm_train, P, Q, K, E, steps, alpha, beta)
            #nR = numpy.dot(nP, nQ)

    
            nP = pickle.load(open("../Experiments/Facebook_MF/P/"+str(E)+"_P60k_k"+str(K)+"_alpha"+str(alpha)+"_beta"+str(beta)+"_first_90train", "rb"))
            nQ = pickle.load(open("../Experiments/Facebook_MF/Q/"+str(E)+"_Q60k_k"+str(K)+"_alpha"+str(alpha)+"_beta"+str(beta)+"_first_90train", "rb"))
    
            nR = numpy.dot(nP, nQ)

            rmse1 = rmse(nR, R_norm_train)
            print(E, rmse1)

            # print ('Cross-Validation Step ' + str(counter) + ' for test' + ":\t" + str(rmse1))
            rmse2 = rmse(nR, R_norm_test)

            print(E, rmse2)

            #mae1 = 0
            #mae2 = 0
            mae1 = mae(nR, train_data_matrix)

            mae2 = mae(nR, test_data_matrix)

            '''
            relv_mask = []
            train_mask2 = []
            nR_mask = []
            for rr in test_data_matrix:
                ts = [i for i in range(len(rr)) if rr[i] > 0]
                relv_mask.append(ts)
            for rr in train_data_matrix:
                ts = [i for i in range(len(rr)) if rr[i] > 0]
                train_mask2.append(ts)
            for rr in nR:
                ts = [i for i in range(len(rr)) if rr[i] > 0]  # i not in train_mask2[i]]
                nR_mask.append(ts)

            relv_mask1 = relv_mask[::]
            for i in range(train_data_matrix.shape[0]):
                for j in range(train_data_matrix.shape[1]):
                    if train_data_matrix[i][j] == 0:
                        relv_mask1[i] = numpy.append(relv_mask1[i], j)
            '''
            MEP = 0
            MER = 0
            #MEP = calculate_MEP(nR, W, relv_mask1, train_mask2, 10)
            #MER = calculate_MER(nR, W, relv_mask1, train_mask2, 10)

            if (MEP + MER) == 0:
                EF1 = 0
            else:
                EF1 = 2 * (MEP * MER) / (MEP + MER)

            # MAP2 = calculate_MAP(R_norm_test,nR, 50)

            # print('MAP:')
            # print(MAP2)
            tp = 0
            fp = 0
            tpAll = []
            fpAll = []

            tn = 0
            tnAll = []
            fn = 0

            etp = 0
            efp = 0
            etn = 0
            efn = 0

            fnAll = []
            precision = 0
            recall = 0
            eprecision = 0
            erecall = 0
            MAP = 0
            MAR = 0
            F1 = 0
            AP = 0
            AR = 0
            MEP2 = 0
            MER2 = 0
            tpr = []
            fpr = []
            y_true = []
            y_score = []

            step = 50
            AUC = []
            AUCC = 0

            """
            """
            AUCC = []
            mAP = []
            # MAPP = []
            maxTopN = 10
            users_num = len(test_data_matrix)
            # users_num = 50
            """
            for u in range(users_num):  # for all users
                # print(u)
                auc = 0
                map1 = 0
                FPR_pre = 0
                TPR_pre = 0
                for st in range(1, maxTopN):
                    # for st in range(50, maxTopN, step):  # starting from top-10 maxTopN = rate.shape[1]
                    # for st in range(1):  # starting from top-10 maxTopN = rate.shape[1]

                    Pred_ind = numpy.argsort(nR[u][1:])[::-1]  # decreasing sort

                    TP = 0
                    FN = 0
                    TN = 0
                    for i, v in enumerate(Pred_ind):
                        if i < st:
                            if v in relv_mask1[u]:
                                TP += 1
                        else:
                            if v in relv_mask1[u]:
                                FN += 1
                            else:
                                TN += 1
                    TPR = float(TP) / (TP + FN)  # recall
                    FP = st - TP
                    FPR = float(FP) / (FP + TN)
                    auc += TPR * (FPR - FPR_pre)
                    FPR_pre = FPR

                    # print TPR,FPR

                    prec = float(TP) / st
                    map1 += prec * (TPR - TPR_pre)
                    TPR_pre = TPR
                    # print prec,TPR
                AUC.append(auc)
                mAP.append(map1)

            AUCC = numpy.mean(AUC)
            MAP = numpy.mean(mAP)
"""

            ord_pred_all = []
            target_all = []
            AUCCCC = 0
            ndcggg = 0
            app = 0
            for u in range(len(test_data_matrix)):
                ord_pred = topnRmap(nR, 50, u)
                target = topnTmap(test_data_matrix, u)
                ord_pred_all.append(ord_pred)
                target_all.append(target)
                M, N = nR.shape
                # AUCC = sum(auc(ord_pred, target) for u in range(M)) / M
                # ndcg = sum(ndcg_k(ord_pred, target) for u in range(M)) / M
                #AUCCC = 0
                AUCCC = auc(ord_pred, target)
                # ndcgg = ndcg_k(topnR(nR, 10, R_norm_test, u), topnT(nR, 10, R_norm_test, u))
                #ndcgg = 0
                ndcgg = ndcg_k(ord_pred, target, 50)

                AUCCCC += AUCCC

                ndcggg += ndcgg

                # ap = ap_k(ord_pred, target, 10)
                # app = app + ap
                # appp = app / M
                # print('User: ', u, 'ord_pred', ord_pred, 'target', target, 'Precision', ap,'ndcg ',ndcgg, 'AUC ', AUCCC, hits)
            # print('appp ', appp)

            AUCC = AUCCCC / M
            ndcg = ndcggg / M

            #MAP = 0
            #MAR = 0
            #F1 = 0
            #ndcg = 0
            #AUCC = 0
            MAP = map_k(ord_pred_all, target_all, 10)
            # AUCC = auc(ord_pred, target)
            # ndcg = ndcg_k(ord_pred, target, 10)
            # print('MAPPP: ',MAPPP)
            # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
            # auc = metrics.auc(fpr,tpr)

            rmse1_E = rmse1_E + rmse1
            rmse2_E = rmse2_E + rmse2

            mae1_E = mae1_E + mae1
            mae2_E = mae2_E + mae2
            MEP_E = MEP_E + MEP
            MER_E = MER_E + MER
            EF1_E = EF1_E + EF1
            MAP_E = MAP_E + MAP
            MAR_E = MAR_E + MAR
            F1_E = F1_E + F1
            ndcg_E = ndcg_E + ndcg
            auc_E = auc_E + AUCC

            print('RMSE Train: ', "%.4f" % rmse1, "%.4f" % rmse1_E)
            print('RMSE Test: ', "%.4f" % rmse2, "%.4f" % rmse2_E)
            print('MAE Train: ', "%.4f" % mae1, "%.4f" % mae1_E)
            print('MAE Test: ', "%.4f" % mae2, "%.4f" % mae2_E)
            print('MEP: ', "%.4f" % MEP, "%.4f" % MEP_E)
            print('MER: ', "%.4f" % MER, "%.4f" % MER_E)
            print('EF1: ', "%.4f" % EF1, "%.4f" % EF1_E)
            print('MAP: ', "%.10f" % MAP, "%.10f" % MAP_E)
            print('MAR: ', "%.4f" % MAR, "%.4f" % MAR_E)
            print('F1: ', "%.4f" % F1, "%.4f" % F1_E)
            print('nDCG: ', "%.10f" % ndcg, "%.10f" % ndcg_E)
            print('AUC: ', "%.10f" % AUCC, "%.10f" % auc_E)

        rmse1m = rmse1_E / E
        rmse2m = rmse2_E / E
        mae1m = mae1_E / E
        mae2m = mae2_E / E
        MEPm = MEP_E / E
        MERm = MER_E / E
        EF1m = EF1_E / E
        MAPm = MAP_E / E
        MARm = MAR_E / E
        F1m = F1_E / E
        ndcgm = ndcg_E / E
        aucm = auc_E / E

        if t:
            eval.append(
                "Metrics" + "," + "K" + "," + "Train RMSE" + "," + "Test RMSE" + "," + "Train MAE" + "," + "Test MAE" + "," + "MEP" + "," + "MER" + "," + "MAP" + "," + "MAR" + "," + "AUC" + "," + "nDCG" + "," + "F1-score" + "," + "EF1-score")
            t = False
        eval.append("[Facebook_MF" + "," + str(
            K) + "," + "%.4f" % rmse1m + "," + "%.4f" % rmse2m + "," + "%.4f" % mae1m + "," + "%.4f" % mae2m + "," + "%.4f" % MEPm + "," + "%.4f" % MERm + "," + "%.10f" % MAPm + "," + "%.4f" % MARm + "," + "%.10f" % aucm + "," + "%.10f" % ndcgm + "," + "%.4f" % F1m + "," + "%.4f" % EF1m)

    evaluation = pandas.DataFrame(eval, columns=['eval'])
    # evaluation.to_csv('../Experiments/MF/Results/MF_Evaluation_first90train_a0.01_b0.1.csv', sep='\n', header=False, float_format='%.2f', index=False, )
    evaluation.to_csv("../Experiments/Facebook_MF/Results/Facebook_MF_Evaluation_first90train.csv", sep='\n', header=False, float_format='%.2f', index=False, )

