import numpy as np
import pandas as pd
import time
import datetime
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def data_clean():
    start_time=time.time()
    # 读取数据
    train=pd.read_csv('../data/meinian_round1_train_20180408.csv',sep=',',encoding='gbk')
    test=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',sep=',',encoding='gbk')
    try:
        data_part1=pd.read_csv('../data/meinian_round1_data_part1_20180408.txt',sep='$',encoding='utf-8')
        data_part2=pd.read_csv('../data/meinian_round1_data_part2_20180408.txt',sep='$',encoding='utf-8')
    except:
        import zipfile
        with zipfile.ZipFile('../data/meinian_round1_data_part1_20180408.zip', 'r') as z:  
            f = z.open('meinian_round1_data_part1_20180408.txt')  
            data_part1 = pd.read_csv(f,sep='$',encoding='utf-8')
        with zipfile.ZipFile('../data/meinian_round1_data_part2_20180408.zip', 'r') as z:  
            f = z.open('meinian_round1_data_part2_20180408.txt')  
            data_part2 = pd.read_csv(f,sep='$',encoding='utf-8')
    
    # data_part1和data_part2进行合并，并剔除掉与train、test不相关vid所在的行
    part1_2 = pd.concat([data_part1,data_part2],axis=0)#{0/'index', 1/'columns'}, default 0
    part1_2 = pd.DataFrame(part1_2).sort_values('vid').reset_index(drop=True)
    vid_set=pd.concat([train['vid'],test['vid']],axis=0)
    vid_set=pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
    part1_2=part1_2[part1_2['vid'].isin(vid_set['vid'])]
    
    # 根据常识判断无用的'检查项'table_id，过滤掉无用的table_id
    def filter_None(data):
        data=data[data['field_results']!='']
        data=data[data['field_results']!='未查']
        return data
    
    part1_2=filter_None(part1_2)
    
    # 过滤列表，过滤掉不重要的table_id 所在行
    filter_list=['0203','0209','0702','0703','0705','0706','0709','0726','0730','0731','3601',
                 '1308','1316']
    
    part1_2=part1_2[~part1_2['table_id'].isin(filter_list)]
    
    # 重复数据的拼接操作
    def merge_table(df):
        df['field_results'] = df['field_results'].astype(str)
        if df.shape[0] > 1:
            merge_df = " ".join(list(df['field_results']))
        else:
            merge_df = df['field_results'].values[0]
        return merge_df
    
    # 数据简单处理
    print(part1_2.shape)
    vid_tabid_group = part1_2.groupby(['vid','table_id']).size().reset_index()
    # print(vid_tabid_group.head())
    # print(vid_tabid_group.shape)
    #                      vid               table_id  0
    # 0  000330ad1f424114719b7525f400660b     0101     1
    # 1  000330ad1f424114719b7525f400660b     0102     3
    
    # 重塑index用来去重,区分重复部分和唯一部分
    print('------------------------------去重和组合-----------------------------')
    vid_tabid_group['new_index'] = vid_tabid_group['vid'] + '_' + vid_tabid_group['table_id']
    vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0]>1]['new_index']
    
    # print(vid_tabid_group_dup.head()) #000330ad1f424114719b7525f400660b_0102
    part1_2['new_index'] = part1_2['vid'] + '_' + part1_2['table_id']
    
    dup_part = part1_2[part1_2['new_index'].isin(list(vid_tabid_group_dup))]
    dup_part = dup_part.sort_values(['vid','table_id'])
    unique_part = part1_2[~part1_2['new_index'].isin(list(vid_tabid_group_dup))]
    
    part1_2_dup = dup_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
    part1_2_dup.rename(columns={0:'field_results'},inplace=True)
    part1_2_res = pd.concat([part1_2_dup,unique_part[['vid','table_id','field_results']]])
    
    table_id_group=part1_2.groupby('table_id').size().sort_values(ascending=False)
    table_id_group.to_csv('../data/part_tabid_size.csv',encoding='utf-8')
    
    # 行列转换
    print('--------------------------重新组织index和columns---------------------------')
    merge_part1_2 = part1_2_res.pivot(index='vid',values='field_results',columns='table_id')
    print('--------------新的part1_2组合完毕----------')
    print(merge_part1_2.shape)
    merge_part1_2.to_csv('../data/merge_part1_2.csv',encoding='utf-8')
    print(merge_part1_2.head())
    del merge_part1_2
    
    time.sleep(10)
    print('------------------------重新读取数据merge_part1_2--------------------------')
    merge_part1_2=pd.read_csv('../data/merge_part1_2.csv',sep=',',encoding='utf-8')
    
    # 删除掉一些出现次数低，缺失比例大的字段，保留超过阈值的特征
    def remain_feat(df,thresh=0.9):
        exclude_feats = []
        print('----------移除数据缺失多的字段-----------')
        print('移除之前总的字段数量',len(df.columns))
        num_rows = df.shape[0]
        for c in df.columns:
            num_missing = df[c].isnull().sum()
            if num_missing == 0:
                continue
            missing_percent = num_missing / float(num_rows)
            if missing_percent > thresh:
                exclude_feats.append(c)
        print("移除缺失数据的字段数量: %s" % len(exclude_feats))
        # 保留超过阈值的特征
        feats = []
        for c in df.columns:
            if c not in exclude_feats:
                feats.append(c)
        print('剩余的字段数量',len(feats))
        return feats
    feats=remain_feat(merge_part1_2,thresh=0.96)
    
    
    merge_part1_2=merge_part1_2[feats]
    merge_part1_2.to_csv('../data/merge_part1_2.csv')
    
    # 找到train，test各自属性进行拼接
    train_of_part=merge_part1_2[merge_part1_2['vid'].isin(train['vid'])]
    test_of_part=merge_part1_2[merge_part1_2['vid'].isin(test['vid'])]
    
    train=pd.merge(train,train_of_part,on='vid')
    test=pd.merge(test,test_of_part,on='vid')
    
    # 清洗训练集中的五个指标
    def clean_label(x):
        x=str(x)
        if '+' in x:#16.04++
            i=x.index('+')
            x=x[0:i]
        if '>' in x:#> 11.00
            i=x.index('>')
            x=x[i+1:]
        if len(x.split(sep='.'))>2:#2.2.8
            i=x.rindex('.')
            x=x[0:i]+x[i+1:]
        if '未做' in x or '未查' in x or '弃查' in x:
            x=np.nan
        if str(x).isdigit()==False and len(str(x))>4:
            x=x[0:4]
        return x
    
    # 数据清洗
    def data_clean(df):
        for c in ['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']:
            df[c]=df[c].apply(clean_label)
            df[c]=df[c].astype('float64')
        return df
    train=data_clean(train)
    
    print('---------------保存train_set和test_set---------------------')
    train.to_csv('../data/train_set.csv',index=False,encoding='utf-8')
    test.to_csv('../data/test_set.csv',index=False,encoding='utf-8')
    
    end_time=time.time()
    print('程序总共耗时:%d 秒'%int(end_time-start_time))    

def feature_engineer(padmethod = 'mean'):
    def Check(Seri,key_name):
        ind = np.zeros([Seri.shape[0]])
        for i in key_name:
            ind +=Seri.str.contains(i,na=-1)
        return np.sign(ind)+1
    
    def mapvalue(x):
        if x==True:
            y = 0
        elif x==False:
            y = 1
        else:
            y = 2
        return y
    
    train = pd.read_csv('../data/train_set.csv',encoding='utf-8')
    test = pd.read_csv('../data/test_set.csv',encoding='utf-8')
    
    merge = pd.concat([train,test],axis = 0)
    label_name = ['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']
    feature_name = [ f for f in train.columns if f not in ['vid','收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']]
    train_num = train.shape[0]
    # 处理标签
    label = merge[label_name].fillna(method='pad')  
    y = label['舒张压']
    y[y>155] = 155
    label['舒张压'] = y    
    # 获取数值型和文本型特征名
    Numerical_feature =[f for f in merge.describe().columns if f not in label_name]
    
    # 文本特征名字
    Character_feature = []
    for i in [f for f in feature_name if f not in Numerical_feature]: 
        try :
            temp = merge[i] 
            temp = temp.fillna('NAN')
            Len = temp.map(len)
            Character_feature.append(i)
        except:
            Numerical_feature.append(i)
      
               
    # 数值型特征处理
    Numeric = merge[Numerical_feature].apply(pd.to_numeric, errors='corece')
    if padmethod == 'mean':
        Numeric = Numeric.fillna( Numeric.mean())
    else:
        Numeric = Numeric.fillna( 0 )
    print('数值型特征{0}'.format(Numeric.shape))
    
    Chrac = pd.DataFrame()
    # 文本型特征处理I
    for i in Character_feature:
        print('process ',i,'loading...')
        temp = merge[i]
        Chrac[i] = Check(temp,['形态大小正常','形态正常','未发现明显异常','未见异常','未见明显'])  
    print('文本型特征I{0}'.format(Chrac.shape)) 
    
    # 文本型特征处理II
    for i in Character_feature:  
        print('process ',i,'loading...')
        temp = merge[i]    
        ind = temp.str.contains(r'[1-9]\d*',na = 'NaN' )
        ind = ind.map(mapvalue)
        temp = temp.fillna('NaN')
        Len = temp.map(len)
        aafeature = pd.concat([ind,Len],axis=1)
        temp.columns = [i+'Number',i+'LenChra']
        Chrac = pd.concat([Chrac,aafeature],axis=1) 
    print('文本型特征I+II{0}'.format(Chrac.shape))   
    
    # 文本型特征处理III
    list1 = ['0102','0409','0434','4001','A202']    
    Key1  = [['脂肪肝',],['脂肪肝','高血压','糖尿病',],['脂肪肝','高血压','糖尿病','冠心病','血糖偏高','血脂偏高'],['硬化','减弱','稍硬'],['脂肪肝','钙化']]
    List1_fea = pd.DataFrame()
    for j,i in enumerate(list1):
        print('process ',i,'loading...')
        temp = Check(merge[i],Key1[j])
        List1_fea = pd.concat( [List1_fea,temp],axis = 1 )
    print('文本型特征III{0}'.format(List1_fea.shape))
    
    
    result = pd.concat([merge['vid'],label,Numeric,Chrac,List1_fea],axis = 1)
    result = result.fillna(result.mean())
    print('所有数据{0}'.format(result.shape))
    
    result.iloc[:train_num,:].to_csv('../data/train_set_cleaned_pad_{0}.csv'.format(padmethod),index=False,encoding='utf-8')
    result.iloc[train_num:,:].to_csv('../data/test_set_cleaned_pad_{0}.csv'.format(padmethod),index=False,encoding='utf-8')

def kfold_cv_lightgbm(padmethod = 'mean'):
    def evalerror(pred, df):
        label = df.get_label().values.copy()
        score = np.mean( ( np.log(pred+1) - np.log(label+1)) **2 )
        return ('log1loss',score,False)
     
    def feval(y_true,y_pred):
        return np.mean( ( np.log(y_pred+1) - np.log(y_true+1)) **2 )       
    # 读取数据
    train_ori = pd.read_csv(r'../data/train_set_cleaned_pad_{0}.csv'.format(padmethod),encoding='utf-8')
    test = pd.read_csv(r'../data/test_set_cleaned_pad_{0}.csv'.format(padmethod),encoding='utf-8')
    label_name = ['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']
    feature_name = [f for f in train_ori.columns if f not in ['vid','收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']]
    # 整理标签中得异常值
    y_trian_true = train_ori[label_name].as_matrix()
    for i in range(5):
        y_tmp = y_trian_true[:,i]
        y_tmp[y_tmp<=0] = np.mean(y_tmp)
        y_trian_true[:,i] = y_tmp
    train_ori[label_name] = y_trian_true
    
    params1 = {
        'learning_rate': 0.005,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'sub_feature': 0.35,
        'num_leaves': 150,
        'subsample_freq':1,
        'subsample':0.7,
        'min_hessian': 1,
        'lambda_l1':0,
        'lambda_l2':0,
        'verbose': -1,
    }
    params2 = {
        'learning_rate': 0.005,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'sub_feature': 0.35,
        'num_leaves': 150,
        'subsample_freq':1,
        'subsample':0.7,
        'min_hessian': 1,
        'lambda_l1':0,
        'lambda_l2':0,
        'verbose': -1,
    }
    params3 = {
        'learning_rate': 0.005,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'sub_feature': 0.35,
        'num_leaves': 200,
        'subsample_freq':1,
        'subsample':0.7,
        'min_hessian': 1,
        'lambda_l1':0,
        'lambda_l2':0,
        'verbose': -1,
    }
    params4 = {
        'learning_rate': 0.005,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'sub_feature': 0.35,
        'num_leaves': 150,
        'subsample_freq':1,
        'subsample':0.7,
        'min_hessian': 1,
        'lambda_l1':0,
        'lambda_l2':0,
        'verbose': -1,
    }
    params5 = {
        'learning_rate': 0.005,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'sub_feature': 0.4,
        'num_leaves': 150,
        'subsample_freq':1,
        'subsample':0.7,
        'min_hessian': 1,
        'lambda_l1':0,
        'lambda_l2':0,
        'verbose': -1,
    }
    params = [params1,params2,params3,params4,params5]
    # 
    n_kf = 10
    
    
    y_test = np.zeros([test.shape[0],5])
    y_valid = np.zeros([train_ori.shape[0],5])
    
    for ln in range(5):
        kf = KFold(n_splits=n_kf,shuffle=True,random_state=233).split(train_ori)
        for j,(train_index,test_index) in enumerate(kf):    
            lgb_train = lgb.Dataset( train_ori[feature_name].iloc[train_index,:],train_ori[label_name[ln]].iloc[train_index])
            lgb_valid = lgb.Dataset(train_ori[feature_name].iloc[test_index,:], train_ori[label_name[ln]].iloc[test_index])
            gbm = lgb.train(params[ln],
                            lgb_train,
                            num_boost_round=5000,
                            valid_sets=lgb_valid,
                            verbose_eval=500,
                            feval=evalerror,
                            early_stopping_rounds=100)
            y_valid[test_index,ln] = gbm.predict(train_ori[feature_name].iloc[test_index])
            y_test[:,ln] += gbm.predict(test[feature_name])
            del gbm
            
    y_test /= n_kf
    
    E = 0
    for i in range(5) :
        e = feval(y_trian_true[:,i], y_valid[:,i]) 
        print('{0}的最终得分为{1}'.format(label_name[i],e))
        E += e
    print('{0}折交叉验证最终得分为{1}'.format(n_kf,E/5))  
    
    
    # 保存结果
    Results = pd.DataFrame( data = y_test, columns = label_name)
    Results = pd.concat( [test['vid'],Results],axis = 1 )    
    if np.all( y_test>0 ):
        Results.to_csv('../data/result_test_{0}.csv'.format(padmethod ),index = False,header=True  )
    else:
        print('结果中包含负值，请检查')
    
    valid_result = pd.DataFrame( data = y_valid, columns = label_name)
    valid_result = pd.concat([train_ori['vid'],valid_result],axis=1)
    valid_result.to_csv('../data/result_train_{0}.csv'.format(padmethod ),index = False,header=True  )    

def model_stack():
    def feval(y_true,y_pred):
        return np.mean( ( np.log(y_pred+1) - np.log(y_true+1)) **2 )
    
    # reading data
    train1 = pd.read_csv('../data/result_train_mean.csv',encoding='ANSI')
    test1 = pd.read_csv('../data/result_test_mean.csv',encoding='ANSI')

    train2 = pd.read_csv('../data/result_train_zero.csv',encoding='ANSI')
    test2 = pd.read_csv('../data/result_test_zero.csv',encoding='ANSI')
    test2.columns = ['vid','收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']
   
    label_name = ['收缩压','舒张压','血清甘油三酯','血清高密度脂蛋白','血清低密度脂蛋白']
    train = pd.read_csv('../data/train_set_cleaned_pad_mean.csv',encoding='utf-8')[label_name]
    
    y_train_true = train[label_name].as_matrix()
    for i in range(5):
        y_tmp = y_train_true[:,i]
        y_tmp[y_tmp<=0] = np.mean(y_tmp)
        y_train_true[:,i] = y_tmp
    train[label_name] = y_train_true
       
    n_kf = 20
    E = 0
    y_train_pred = np.zeros([y_train_true.shape[0],5])
    y_test = np.zeros([test1.shape[0],5])
    print('======================stack模型训练开始===============================')
    for i,ln in enumerate(label_name):
        merge_train = pd.concat( [train1[ln],train2[ln]],axis=1)
        merge_test = pd.concat( [test1[ln],test2[ln]],axis=1)
        kf = KFold(n_splits=n_kf,shuffle=True,random_state=233).split(merge_train)
        print('==============={0}{1}折交叉验证开始======================'.format(ln,n_kf))
        for j,(train_index,test_index) in enumerate(kf):
            print('第{0}折训练开始'.format(j+1))
            train_set = merge_train.iloc[train_index,:]
            valid_set = merge_train.iloc[test_index,:]
            rgs = LinearRegression()
            rgs.fit(train_set,train[ln].iloc[train_index])
            y_pr = rgs.predict(valid_set)
            print('第{0}折训练完成,得分{1}'.format(j+1,feval(train[ln].iloc[test_index],y_pr)))
            y_train_pred[test_index,i] = y_pr
            y_test[:,i] += rgs.predict( merge_test )
            del rgs
        sc = feval(train[ln],y_train_pred[:,i])
        print('{0}训练结束,得分{1}'.format(ln,sc))
        E += sc
    print('stack模型训练完毕,最终得分{0}'.format(E/5))
    y_test /= n_kf 
       
    result = pd.DataFrame(y_test,columns = label_name)
    result = pd.concat( [test1['vid'],result],axis = 1 )    
    #
    if np.all( y_test>0 ):
        result.to_csv('../submit/submit_{0}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),index = False,header=False  )
    else:
        print('结果中包含负值，请检查')    

if __name__ == '__main__'    :
    # 数据清洗 该部分代码 根据 技术圈Jean_V 文章进行修改
    data_clean()
    # 特征提取 缺省值分别以 均值 和 0 代替
    feature_engineer(padmethod='mean')  
    feature_engineer(padmethod='zero')
    # 采用lightgbm 分别对 上述两种特征进行训练
    kfold_cv_lightgbm(padmethod='mean')
    kfold_cv_lightgbm(padmethod='zero')
    # 采用线性回归 对 上述两结果进行 模型融合
    model_stack()

