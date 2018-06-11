from odps.df import DataFrame
import numpy as np
import pandas as pd
import datetime

def feval(y_true,y_pred):
    return np.mean( ( np.log(y_pred+1) - np.log(y_true+1)) **2 )

train_1 = DataFrame(o.get_table('result_valid_for_missth70_pad_mean_gbdt_20180524_192245_3639')).to_pandas()
test_1 = DataFrame(o.get_table('result_for_missth70_pad_mean_gbdt_20180524_192245_3639')).to_pandas()

train_2 = DataFrame(o.get_table('result_valid_for_missth80_pad_mean_gbdt_20180524_181600_3604')).to_pandas()
test_2 = DataFrame(o.get_table('result_for_missth80_pad_mean_gbdt_20180524_181600_3604')).to_pandas()

train_3 = DataFrame(o.get_table('result_valid_for_missth90_pad_mean_gbdt_20180524_165703_3565')).to_pandas()
test_3 = DataFrame(o.get_table('result_for_missth90_pad_mean_gbdt_20180524_165703_3565')).to_pandas()

train_4 = DataFrame(o.get_table('result_valid_for_missth95_pad_mean_gbdt_20180524_170844_3586')).to_pandas()
test_4 = DataFrame(o.get_table('result_for_missth95_pad_mean_gbdt_20180524_170844_3586')).to_pandas()

train_5 = DataFrame(o.get_table('result_valid_for_missth80_pad_zero_gbdt_20180524_193631_3699')).to_pandas()
test_5 = DataFrame(o.get_table('result_for_missth80_pad_zero_gbdt_20180524_193631_3699')).to_pandas()

train_6 = DataFrame(o.get_table('result_valid_for_missth90_pad_zero_gbdt_20180524_194536_3751')).to_pandas()
test_6 = DataFrame(o.get_table('result_for_missth90_pad_zero_gbdt_20180524_194536_3751')).to_pandas()

train_7 = DataFrame(o.get_table('result_valid_for_gbdt_20180522_215402_3549')).to_pandas()
test_7 = DataFrame(o.get_table('result_for_gbdt_20180522_215357_3549')).to_pandas()

train_8 = DataFrame(o.get_table('result_valid_for_gbdt_20180522_213611_3591')).to_pandas()
test_8 = DataFrame(o.get_table('result_for_gbdt_20180522_213606_3591')).to_pandas()

from sklearn.linear_model import LinearRegression,Ridge
from sklearn.svm import LinearSVR
from sklearn.model_selection import KFold

label_name = ['sys','dia','tl','hdl','ldl']

def stack_cell(rgs,n_kf,ln,y_true,*data):
    train = pd.DataFrame()
    test = pd.DataFrame()
    for i,(d1,d2) in enumerate(data):
        train = pd.concat([train,d1[ln]],axis = 1)
        test = pd.concat([test,d2[ln]],axis = 1 )
    train = train.as_matrix()
    test = test.as_matrix()
    y_te = np.zeros([test.shape[0]])
    y_pred = np.zeros_like(y_true,dtype='float64')
    kf = KFold(n_splits=n_kf).split(train)
    for j,(train_index,test_index) in enumerate(kf):
        rgs.fit( train[train_index,:], y_true[train_index] )
        y_pred[test_index] = rgs.predict( train[test_index] )
        y_te += rgs.predict( test )
    y_te /= n_kf
    print( '{0} score {1}'.format(ln,feval(y_true,y_pred)))
    return y_pred,y_te,feval(y_true,y_pred)
        

train_true = DataFrame(o.get_table('y_train')).to_pandas()

rgss = [LinearRegression(),
        LinearSVR(C=0.01),
        Ridge(alpha = 1.0)]
rgss_name = ['LR','SVR','Ridge',]

train_s = []
test_s = []
for i,rn in enumerate(rgss_name):
    n_kf = 10
    y_valid = np.zeros([train_1.shape[0],5])
    y_test = np.zeros([test_1.shape[0],5])   
    for j,ln in enumerate(label_name):
        y_true = train_true[ln].as_matrix()
        rgs = rgss[i]
        y_va,y_te,sc = stack_cell(rgs,n_kf,ln,y_true,(train_1,test_1),(train_2,test_2),(train_3,test_3),(train_4,test_4),(train_5,test_5),(train_6,test_6),
                                 (train_7,test_7),(train_8,test_8))
        y_valid[:,j] = y_va
        y_test[:,j] = y_te
    train_s.append( pd.DataFrame(data = y_valid,columns = label_name )  )
    test_s.append( pd.DataFrame(data = y_test,columns = label_name ) )


n_kf = 3
test_s2 = []
fitns = np.zeros([len(rgss_name),5])
for j,rn in enumerate(rgss_name):
    y_test = np.zeros([test_1.shape[0],5])
    for i,ln in enumerate(label_name):
        y_true = train_true[ln].as_matrix()
        rgs = rgss[j]
        y_va,y_te,sc = stack_cell(rgs,n_kf,ln,y_true,(train_1,test_1),(train_2,test_2),(train_3,test_3),(train_4,test_4),(train_5,test_5),(train_6,test_6),
                                  (train_7,test_7),(train_8,test_8),(train_s[0],test_s[0]),(train_s[1],test_s[1]),(train_s[2],test_s[2]) )
        y_test[:,i] = y_te
        fitns[j,i] = sc    
    test_s2.append(y_test)
    
    

y_test = np.zeros([test_1.shape[0],5])
E = 0
for i,ln in enumerate(label_name):
    ind = np.argmin(test_s2[:,i])
    y_test[:,i] = test_s2[ind][:,i]
    print('{0} score {1}'.format(i,np.min(test_s2[:,i])))
    E += np.min(test_s2[:,i])
E /= 5
print ( 'final score {0}'.format(E) )    



result = pd.DataFrame(y_test,columns = label_name)
result = pd.concat( [test_1['vid'],result],axis = 1 )      

if np.all( y_test>0 ):
    table_name = 'result_for_stack_'+str(int(E*100000))
    odps.delete_table(table_name, if_exists=True)
    odps.create_table(table_name, 'vid string, sys bigint, dia bigint, tl double, hdl double, ldl double')
    t = odps.get_table(table_name)
    with t.open_writer() as writer:
        outdata = np.array(result).tolist()
        writer.write(outdata)
        print("outdata length", len(outdata))
    writer.close()
else:
    print('结果中包含负值，请检查')  
