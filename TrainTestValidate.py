# input format:
# user_id, movie_id, rating, time
#
# This code produces 3 .csv files containing:
# 1- Training set --> 80%
# 2- Validating set --> 10% hidden ratings
# 3- Testing set --> 10%

import pandas

c =0
ratings_list_original= [i.strip().split(",") for i in open('../Data/BX-Book-Ratings_AndBookID_Mapped_Sorted_NoZeros_82K_SortedBooks_Norm.csv', 'r').readlines()]
data = pandas.read_csv("../Data/matrix_fatorization_data_indexed_3_normalized.csv", sep=',', names="uid,fid,interactions".split(","))

#data = pandas.read_csv("../Data/matrix_fatorization_data_indexed_3.csv", sep=',', names="uid_index,uid,fid_index,fid,interactions".split(","))
#r_df = pandas.DataFrame(data, columns = ['uid_index', 'fid_index', 'interactions'], dtype=float)



#ratings_df_original = pandas.DataFrame(ratings_list_original, columns=['UserID', 'BookID', 'Rating', 'UserFrequency', 'BookFrequency'], dtype=int)
ratings_df_original = pandas.DataFrame(data, columns=['uid', 'fid', 'interactions'])

#print r_df.shape
print data.shape
print ratings_df_original.shape

#ratings_df_original = pandas.DataFrame(ratings_list_original, columns=['UserID', 'BookID', 'Rating'], dtype=int)


#ratings_list_original = [i.strip().split(",") for i in open('u_data_mapping_NOdublicate_Sorted_Norm.csv', 'r').readlines()]
#ratings_list_original = [i.strip().split(",") for i in open('u_data_mapping_NOdublicate_Sorted_Norm.csv', 'r').readlines()]
#ratings_list_original = [i.strip().split(",") for i in open('u_data_Norm.csv', 'r').readlines()]
#ratings_list_original = [i.strip().split(",") for i in open('u.data', 'r').readlines()]
#ratings_df_original = pandas.DataFrame(ratings_list_original, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
ratings_df_original1 = ratings_df_original.as_matrix()
print len(ratings_df_original1[0,:])
print len(ratings_df_original1[:,0])
m = []
#m1 = []

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
def getTrainSet(c3,i):
    m1=[]
    m2=[]
    m3=[]
    for j in range(len(ratings_df_original1)):
        #print(j)
        if ratings_df_original1[j,0] == i:
            if c3 !=0:
                m1.append(str(ratings_df_original1[j,0])+','+str(ratings_df_original1[j,1])+','+str(0))

                #if c4!=0:
                m2.append(str(ratings_df_original1[j, 0]) + ',' + str(ratings_df_original1[j, 1]) + ',' + str(ratings_df_original1[j,2]))
                #m3.append(str(ratings_df_original1[j, 0]) + ',' + str(ratings_df_original1[j, 1]) + ',' + str(ratings_df_original1[j,2]) + ',' + str(ratings_df_original1[j, 3]))
                c3 = c3 - 1
                #c4=c4-1
                #elif c4 == 0:
                    #m2.append(str(ratings_df_original1[j, 0]) + ',' + str(ratings_df_original1[j, 1]) + ',' + str(ratings_df_original1[j, 2]) + ',' + str(ratings_df_original1[j, 3]))
                    #m3.append(str(ratings_df_original1[j, 0]) + ',' + str(ratings_df_original1[j, 1]) + ',' + str(0) + ',' + str(ratings_df_original1[j, 3]))
            elif c3==0:
                m1.append(str(ratings_df_original1[j,0])+','+str(ratings_df_original1[j,1])+','+str(ratings_df_original1[j,2]))
                m2.append(str(ratings_df_original1[j, 0]) + ',' + str(ratings_df_original1[j, 1]) + ',' + str(0))
                #m3.append(str(ratings_df_original1[j, 0]) + ',' + str(ratings_df_original1[j, 1]) + ',' + str(0) + ',' + str(ratings_df_original1[j, 3]))

            #else:
                #break
        #print(m1)
        #print(m2)
    return m1,m2

d = 0
mm1 = []
mm2 = []
mm3 = []
mm11 = []
mm22 = []
mm33 = []
#print('###########',len(ratings_df_original1))
#print('###########',len(ratings_df_original1[0]))
#usersID = ratings_df_original['uid'].drop_duplicates()
#print usersID,'uid'

n_users = data.uid.unique().shape[0]
n_items = data.fid.unique().shape[0]

print('n_users', n_users)
for i in range(n_users):
    #print('iii',i)
    for j in range(n_items):
        if ratings_df_original1[j,0] == i:
            c = c + 1
    c1 = float(c * 0.1)
    if c1 < 1:
        c1 = int(1)
    else:
        c1 = int(c1)
    #c1 = int(c*0.1)
    #c2 = int(c1*0.5)
    #print(c1)
    mm1,mm2=getTrainSet(c1,i)
    mm11 = mm11 + mm1
    mm22 = mm22 + mm2
    #print(mm11)

    #mm33 = mm33 + mm3
    c1=0
    c2=0
    c=0

m_df = pandas.DataFrame(mm11, columns=['UserID'])
m_df.to_csv('../Data/matrix_fatorization_data_indexed_3_normalized_Train.csv', sep='\n', header=False, float_format='%.2f', index=False,)
#m_df.to_csv('u_data_mapping_NOdublicate_Sorted_Norm_Train_first80train.csv', sep='\n', header=False, float_format='%.2f', index=False,)
#m_df.to_csv('u_data_Train_first80train.csv', sep='\n', header=False, float_format='%.2f', index=False,)

m_df = pandas.DataFrame(mm22, columns=['UserID'])
m_df.to_csv('../Data/matrix_fatorization_data_indexed_3_normalized_Test.csv', sep='\n', header=False, float_format='%.2f', index=False,)
#m_df.to_csv('u_data_Validate_first80train.csv', sep='\n', header=False, float_format='%.2f', index=False,)

#m_df = pandas.DataFrame(mm33, columns=['UserID'])
#m_df.to_csv('u_data_mapping_NOdublicate_Sorted_Norm_Validate_first90train_V.csv', sep='\n', header=False, float_format='%.2f', index=False,)
#m_df.to_csv('u_data_mapping_NOdublicate_Sorted_Norm_Test_first80train.csv', sep='\n', header=False, float_format='%.2f', index=False,)
#m_df.to_csv('u_data_Test_first80train.csv', sep='\n', header=False, float_format='%.2f', index=False,)

