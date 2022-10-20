


import pandas

data = pandas.read_csv("../Data/matrix_fatorization_data_indexed_3.csv", sep=',', names="uid_index,uid,fid_index,fid,interactions".split(","))
r_df = pandas.DataFrame(data, columns = ['uid_index', 'fid_index', 'interactions'], dtype=float)

print max(data['interactions'])

#r = [i.strip().split(",") for i in open('../Data/BX-CSV-Dump/BX-Book-Ratings_AndBookID_Mapped_Sorted_NoZeros_82K_SortedBooks.csv', 'r').readlines()]

#r_df = pandas.DataFrame(r, columns = ['UserID', 'BookID', 'Rating'], dtype=float)

#r1 = r_df
#r2 = r1.Rating/10
#r1 = r1['Rating']/10
m = []
for i in range(len(r_df)):
    print(i)
    rating = round(float(r_df['interactions'][i]/max(data['interactions'])), 2)
    m.append(str(int(r_df['uid_index'][i]))+','+str(int(r_df['fid_index'][i]))+','+str(float(rating)))



m_df = pandas.DataFrame(m, columns=['interactions'])

print(len(m_df))

m_df1 = m_df.drop_duplicates()

print(len(m_df1))

m_df1.to_csv('../Data/matrix_fatorization_data_indexed_3_normalized.csv', sep=',', header=False, float_format='%.4f', index=False,)





#R_df_original = ratings_df_original.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)


#R_original = R_df_original.as_matrix()


#R_norm_original = (R_original - R_original.min()) / (R_original.max() - R_original.min())
