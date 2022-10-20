"""


This code maps the original ratings to the mapped books resulting in a lower number of ratings.

"""


import pandas
import numpy


id= [i.strip().split(",") for i in open('../Data/matrix_fatorization_data.csv', 'r').readlines()]
id_df = pandas.DataFrame(id, columns=['uid', 'fid', 'interactions'])

#print(id_df['uid'])

u = id_df['fid'].drop_duplicates()

print len(u)

#my_array = numpy.array(u)
#my_array1 = my_array.sort()
#u = sorted(u)

#print(my_array1)

"""

m = []
d = []
c=0
print(len(id_df))
print(len(noid_df))

u = noid_df['BookID'].drop_duplicates()
u = sorted(u)

print(len(u))

moviemapwithID_df = pandas.DataFrame(u, columns=['BookID'])
moviemapwithID_df.to_csv('../Data/BX-CSV-Dump/BX-Books_Only.csv', sep='\n', header=False, float_format='%.2f', index=False,)

"""


"""
for i in range(len(noid_df)):
    print(i)
    if noid_df['ISBN'][i] not in id_df['ISBN']:
        noid_df.drop([i])
print(len(noid_df))
print(noid_df)
"""

"""
for i in range(len(id_df)):
    print(i)
    for j in range(len(noid_df)):
        if id_df['ISBN'][i] == noid_df['ISBN'][j]:
            m.append(str(noid_df['UserID'][j]).strip('/"') +'||'+ str(id_df['BookID'][i]) +'||'+ str(noid_df['ISBN'][j]) +'||'+ str(noid_df['Rating'][j]).strip('/"'))

print len(m)



moviemapwithID_df = pandas.DataFrame(m, columns=['MovieID'])
moviemapwithID_df.to_csv('../Data/BX-CSV-Dump/BX-Users_60K.csv', sep='\n', header=False, float_format='%.2f', index=False,)

#print moviemapwithID_df
"""

"""

movies_list = [i.strip().split("(") for i in open('ml-100k/u_csv.csv', 'r').readlines()]
movies_df = pandas.DataFrame(movies_list, columns=['MovieName', 'Year', 'G'])
del movies_df['Year']
del movies_df['G']
movies_df.to_csv('outfile_movielens.dat', sep='\n', header=False, float_format='%.2f', index=False)


m = movies_df.as_matrix()
#print m[2]
"""
#arr = numpy.array(movies_df)
#print arr
#pickle.dump(m, open("M_list.dat", "wb"))
#print (movies_df)
#newlist1 =[]
#for i in range(len(movies_list)):
 #   newlist1.append(movies_list[i])
#    print (newlist1[i])
#print len(newlist1)