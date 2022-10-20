import numpy
import pandas

data = pandas.read_csv("../Data/matrix_fatorization_data_indexed1.csv", sep=',', names="uid_index,uid,fid_index,fid,interactions".split(","))
n_users = data.uid_index.unique().shape[0]
n_items = data.fid_index.unique().shape[0]

#ratings_list_original = [i.strip().split(",") for i in open('../Data/matrix_fatorization_data_indexed1.csv', 'r').readlines()]
#ratings_df_original = pandas.DataFrame(ratings_list_original, columns=['uid_index', 'uid', 'fid_index', 'fid', 'interactions'],dtype=int)
#data = ratings_df_original.pivot(index = 'uid', columns ='fid', values = 'interactions').fillna(0)
#data1 = data.as_matrix()

#print R_original


#print n_users
#print n_items

#print data

data1 = data['uid_index'].drop_duplicates() # result is 6427 unique uid

data2 = data['fid_index'].drop_duplicates() # result is 8639 unique fid

print(data1)
print(data2)


#data11 = [(idx, item) for idx,item in enumerate(data1, start=1)] # here I successfully give index for each uid starting from 1

#data22 = [(idx, item) for idx,item in enumerate(data2, start=1)] # here is the same but the problem is we have 2 inedeces for the same uid and fid
n = []
d = 0
for j in data2:

    # print i
    for k in data1:
        # print j
        if j == k:
            d = d + 1
            n.append(k)
print 'dddddddddddddddddddddddddddd',d
print n

m = []
c = 0
for i in range(1, len(data2)+1):
    for j in data2:

        #print i
        #for k in data1:
            #print j
        if j == i:
            c = c + 1
            print 'j',j
            m.append(j)

            #break
        '''
        if i in m:
            print 'i',i
            i = i+1
            break
        else:
            m.append(i)

            break
        '''
    #break

                #print i, j

print m
print 'Total number of shared ids between uid and fid is: '
print c

'''
############################
# We have 3896 shared ids between uid and fid
###########################

#print data22

from sklearn import cross_validation as cv

train_data, test_data = cv.train_test_split(data, test_size=0.1)
print 'shapeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
print train_data.shape
print test_data.shape
# Create two user-item matrices, one for training and another for testing
train_data_matrix = numpy.zeros((data.shape))
for line in train_data.itertuples():
    train_data_matrix[int(line[1]) - 1, int(line[3]) - 1] = int(line[5])
    #print train_data_matrix
test_data_matrix = numpy.zeros((data.shape))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[3] - 1] = line[5]

print train_data_matrix.shape
#print test_data_matrix.shape

'''