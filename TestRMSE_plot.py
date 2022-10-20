import matplotlib.pyplot as plt
import pandas

alpha = 0.01
beta = 0.001
K = 5


Facebook_MF_Evaluation = [i.strip().split(",") for i in open("../Experiments/Facebook_MF/Results/"+"10_P60k_k"+str(K)+"_alpha"+str(alpha)+"_beta"+str(beta)+"_first_90train.csv", 'r').readlines()]
Facebook_MF_Evaluation_df = pandas.DataFrame(Facebook_MF_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

Facebook_PMF_Evaluation = [i.strip().split(",") for i in open("../Experiments/Facebook_PMF/Results/Facebook_PMF_Evaluation_first90train.csv", 'r').readlines()]
Facebook_PMF_Evaluation_df = pandas.DataFrame(Facebook_PMF_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

"""
EMF_Evaluation = [i.strip().split(",") for i in open('../Experiments/EMF/Results/EMF_Evaluation_first90train.csv', 'r').readlines()]
EMF_Evaluation_df = pandas.DataFrame(EMF_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)


SemEMF_Evaluation = [i.strip().split(",") for i in open('../Experiments/SemEMF/Results/SemEMF_Evaluation_first90train.csv', 'r').readlines()]
SemEMF_Evaluation_df = pandas.DataFrame(SemEMF_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

MergedSemEMF_Evaluation = [i.strip().split(",") for i in open('../Experiments/MergedSemEMF/Results/MergedSemEMF_Evaluation_first90train.csv', 'r').readlines()]
MergedSemEMF_Evaluation_df = pandas.DataFrame(MergedSemEMF_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

ASEMFI_Evaluation = [i.strip().split(",") for i in open('../Experiments/ASEMFI/Results/ASEMFI_Evaluation_first90train.csv', 'r').readlines()]
ASEMFI_Evaluation_df = pandas.DataFrame(ASEMFI_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

ASEMFU_Evaluation = [i.strip().split(",") for i in open('../Experiments/ASEMFU/Results/ASEMFU_Evaluation_first90train.csv', 'r').readlines()]
ASEMFU_Evaluation_df = pandas.DataFrame(ASEMFU_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

ASEMF_IB_Evaluation = [i.strip().split(",") for i in open('../Experiments/ASEMF_IB/Results/ASEMF_IB_Evaluation_first90train.csv', 'r').readlines()]
ASEMF_IB_Evaluation_df = pandas.DataFrame(ASEMF_IB_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

ASEMF_UB_Evaluation = [i.strip().split(",") for i in open('../Experiments/ASEMF_UB/Results/ASEMF_UB_Evaluation_first90train.csv', 'r').readlines()]
ASEMF_UB_Evaluation_df = pandas.DataFrame(ASEMF_UB_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

ASEMF_UIB_S_Evaluation = [i.strip().split(",") for i in open('../Experiments/ASEMF_UIB_Subject/Results/ASEMF_UIB_Subject_Evaluation_first90train.csv', 'r').readlines()]
ASEMF_UIB_S_Evaluation_df = pandas.DataFrame(ASEMF_UIB_S_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

ASEMF_UIB_A_Evaluation = [i.strip().split(",") for i in open('../Experiments/ASEMF_UIB_Author/Results/ASEMF_UIB_Author_Evaluation_first90train.csv', 'r').readlines()]
ASEMF_UIB_A_Evaluation_df = pandas.DataFrame(ASEMF_UIB_A_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

LDSDMF_Evaluation = [i.strip().split(",") for i in open('../Experiments/LDSDMF/Results/LDSDMF_Evaluation_first90train.csv', 'r').readlines()]
LDSDMF_Evaluation_df = pandas.DataFrame(LDSDMF_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

LDSDMF_W_Evaluation = [i.strip().split(",") for i in open('../Experiments/LDSDMF_W/Results/LDSDMF_W_Evaluation_first90train.csv', 'r').readlines()]
LDSDMF_W_Evaluation_df = pandas.DataFrame(LDSDMF_W_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)
"""

#R_df_original = ratings_df_original.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
#MF_Evaluation_df[MF_Evaluation_df['MEP@50'][1:]] = MF_Evaluation_df[MF_Evaluation_df['MEP@50'][1:]].convert_objects(convert_numeric = True)
#a = MF_Evaluation_df[MF_Evaluation_df['MEP@50'][1:]]
a = []
b = []
c = []
d = []
e = []
f = []
g = []
h = []
k = []
l = []
m = []
n = []
o = []
p = []

testrmse = 'Test RMSE'

for i in range(1,len(Facebook_MF_Evaluation_df[testrmse])):
    a.append(Facebook_MF_Evaluation_df[testrmse][i])
#print(a)

for i in range(1,len(Facebook_MF_Evaluation_df['K'])):
    k.append(Facebook_MF_Evaluation_df['K'][i])
#print(k)
for i in range(1,len(Facebook_PMF_Evaluation_df[testrmse])):
    l.append(Facebook_PMF_Evaluation_df[testrmse][i])
'''
for i in range(1,len(EMF_Evaluation_df[testrmse])):
    b.append(EMF_Evaluation_df[testrmse][i])



for i in range(1,len(SemEMF_Evaluation_df[testrmse])):
    c.append(SemEMF_Evaluation_df[testrmse][i])

for i in range(1,len(MergedSemEMF_Evaluation_df[testrmse])):
    d.append(MergedSemEMF_Evaluation_df[testrmse][i])

for i in range(1,len(ASEMFI_Evaluation_df[testrmse])):
    e.append(ASEMFI_Evaluation_df[testrmse][i])

for i in range(1,len(ASEMFU_Evaluation_df[testrmse])):
    f.append(ASEMFU_Evaluation_df[testrmse][i])

for i in range(1,len(ASEMF_IB_Evaluation_df[testrmse])):
    g.append(ASEMF_IB_Evaluation_df[testrmse][i])

for i in range(1,len(ASEMF_UB_Evaluation_df[testrmse])):
    h.append(ASEMF_UB_Evaluation_df[testrmse][i])

for i in range(1,len(ASEMF_UIB_S_Evaluation_df[testrmse])):
    m.append(ASEMF_UIB_S_Evaluation_df[testrmse][i])

for i in range(1,len(ASEMF_UIB_A_Evaluation_df[testrmse])):
    n.append(ASEMF_UIB_A_Evaluation_df[testrmse][i])

for i in range(1,len(LDSDMF_Evaluation_df[testrmse])):
    o.append(LDSDMF_Evaluation_df[testrmse][i])

for i in range(1,len(LDSDMF_W_Evaluation_df[testrmse])):
    p.append(LDSDMF_W_Evaluation_df[testrmse][i])

'''

plt.figure()
lw = 1
plt.plot(k,a, 's-' , color='darkorange', lw=lw, label='Facebook_MF')
plt.plot(k,l, 'd-' , color='indianred', lw=lw, label='Facebook_PMF')

'''
plt.plot(k,b, '^-' , color='red', lw=lw, label='EMF')
plt.plot(k,e, '<-' , color='black', lw=lw, label='ASEMFI')
#plt.plot(k,f, '<-' , color='yellow', lw=lw, label='ASEMFU')
#plt.plot(k,g, 'x-' , color='skyblue', lw=lw, label='ASEMF_IB')
#plt.plot(k,h, '--' , color='cyan', lw=lw, label='ASEMF_UB')
plt.plot(k,m, 'o-' , color='blue', lw=lw, label='ASEMF_UIB_S')
#plt.plot(k,n, 'o-' , color='blue', lw=lw, label='ASEMF_UIB_A')
plt.plot(k,c, 'D-' , color='lightgreen', lw=lw, label='SemEMF')
plt.plot(k,d, 'P-' , color='green', lw=lw, label='MergedSemEMF')
plt.plot(k,o, 'd-' , color='darkred', lw=lw, label='LDSDMF')
plt.plot(k,p, '4-' , color='indianred', lw=lw, label='LDSDMF_W')
'''

#plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.5, 5.5])
plt.ylim([0, 0.7])
plt.xlabel('Factors (K)')
plt.ylabel(testrmse)
plt.title(testrmse)
#plt.legend(bbox_to_anchor=(0.25, 1.15), loc=1, borderaxespad=0.)
plt.legend(loc="upper left")
plt.grid()
plt.savefig('Figures/'+str(testrmse)+'.png')
plt.show()