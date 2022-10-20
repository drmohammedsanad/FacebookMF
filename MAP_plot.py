import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas
import numpy

MF_Evaluation = [i.strip().split(",") for i in open('../Experiments/MF/Results/MF_Evaluation_first90train.csv', 'r').readlines()]
MF_Evaluation_df = pandas.DataFrame(MF_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

EMF_Evaluation = [i.strip().split(",") for i in open('../Experiments/EMF/Results/EMF_Evaluation_first90train.csv', 'r').readlines()]
EMF_Evaluation_df = pandas.DataFrame(EMF_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

PMF_Evaluation = [i.strip().split(",") for i in open('../Experiments/PMF/Results/PMF_Evaluation_first90train.csv', 'r').readlines()]
PMF_Evaluation_df = pandas.DataFrame(PMF_Evaluation, columns = ['M','K', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'MEP', 'MER', 'MAP', 'MAR', 'AUC', 'nDCG', 'F1-score', 'EF1-score'], dtype = int)

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

testrmse = 'MAP'

for i in range(1,len(MF_Evaluation_df[testrmse])):
    a.append(MF_Evaluation_df[testrmse][i])

for i in range(1,len(MF_Evaluation_df['K'])):
    k.append(MF_Evaluation_df['K'][i])

for i in range(1,len(EMF_Evaluation_df[testrmse])):
    b.append(EMF_Evaluation_df[testrmse][i])

for i in range(1,len(PMF_Evaluation_df[testrmse])):
    l.append(PMF_Evaluation_df[testrmse][i])

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



plt.figure()
lw = 1
plt.plot(k,a, 's-' , color='darkorange', lw=lw, label='MF(none)')
plt.plot(k,b, '^-' , color='red', lw=lw, label='EMF(N)')
plt.plot(k,l, 'd-' , color='indianred', lw=lw, label='PMF(none)')
plt.plot(k,e, '<-' , color='black', lw=lw, label='AMF(S)')
#plt.plot(k,f, '<-' , color='yellow', lw=lw, label='ASEMFU(S)')
plt.plot(k,m, 'o-' , color='blue', lw=lw, label='ASEMF_UIB(S)')
plt.plot(k,c, 'D-' , color='lightgreen', lw=lw, label='SemEMF(S)')
plt.plot(k,d, 'P-' , color='green', lw=lw, label='MergedSemEMF(S+N)')
#plt.plot(k,g, 'x-' , color='skyblue', lw=lw, label='ASEMF_IB(S)')
#plt.plot(k,h, '--' , color='cyan', lw=lw, label='ASEMF_UB(S)')
#plt.plot(k,q, 'o-' , color='blue', lw=lw, label='ASEMF_UIB_S(S)')

#plt.plot(k,n, 'o-' , color='lightblue', lw=lw, label='ASEMF_UIB_A')
plt.plot(k,o, 'd-' , color='darkred', lw=lw, label='LDSDMF(S)')
#plt.plot(k,p, 'o-' , color='indianred', lw=lw, label='LDSDMF_W(S&N)')

#plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([9, 51])
plt.ylim([0.001, 0.0045])
plt.xlabel('Factors (K)')
plt.ylabel('MAP@10')
plt.title(testrmse+'@10 in Book Domain')
#plt.legend(bbox_to_anchor=(1.13, 1.15), loc=1, borderaxespad=0.)
plt.legend(loc="lower left")
plt.grid()
plt.savefig('Figures/F/'+'Book_'+str(testrmse)+'@10'+'.jpg')
plt.show()
