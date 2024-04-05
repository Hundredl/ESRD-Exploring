'''
process diet data
'''

import datetime
from unittest import result
from loguru import logger
import argparse
import pickle
from typing import Dict,List
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import global_const as gconst
import copy

def diet_dict2dict(diet,no_id=False,no_date=False):
    '''convert diet dict to dict'''
    result_dict = {'id':[]}
    keys = list(diet[list(diet.keys())[0]].keys())
    for key in keys:
        result_dict[key] = []
    
    for id in diet:
        result_dict['id'] += [id]*len(diet[id][keys[0]])
        for key in keys:
            result_dict[key]+=(diet[id][key])
    if no_id:
        result_dict.pop('id')
    if no_date:
        result_dict.pop('date')
    return result_dict


def diet_dict2matrix(diet,no_id=False,no_date=False):
    '''
    change diet dict to numpy array
    '''
    diet_dict = diet_dict2dict(diet,no_id,no_date)
    keys = list(diet_dict.keys())
    result = np.array([])
    for key in keys:
        if len(result) == 0:
            result = np.array(diet_dict[key]).reshape(-1,1)
        else:
            result = np.hstack((result,np.array(diet_dict[key]).reshape(-1,1)))
    # logger.debug(result.shape)
    return result

def output_diet(diet,filename):
    '''output diet to filename(csv)'''
    result_dict = diet_dict2dict(diet)
    with open(filename,'w') as f:
        pd.DataFrame(result_dict).to_csv(f,index=False,header=True)

def get_diet_part(diet:Dict,ids:List) -> Dict:
    '''get part of diet by ids'''
    result = {}
    for id in ids:
        result[id] = diet[id]
    return result

def pic_distribution(diet,filepath,filename,pairplot=False):
    '''draw the distribution of diet'''
    diet_dict = diet_dict2dict(diet,no_id=True,no_date=True)
    save_position = filepath + filename
    if pairplot:
        plt.figure()
        sns.pairplot(pd.DataFrame(diet_dict))
        plt.savefig(filepath + 'diet_distribution_pairplot.png')
    data = {'label':[],'value':[]}
    for key in diet_dict:
        data_one = diet_dict[key]
        data['label'] += [key]*len(data_one)
        data['value'] += data_one
    plt.figure()
    g = sns.FacetGrid(pd.DataFrame(data),col='label',col_wrap=6,sharey=False,sharex=False)
    g.map_dataframe(sns.histplot,x='value')    
    # sns.displot(pd.DataFrame(data),x='value',col='label',col_wrap=6,facet_kws={'sharex':False,'sharey':False})
    plt.savefig(save_position)


def get_ok_diet_for_one(diet_dict,times=1.5):
    '''input diet for one person, return if the data is ok, the reason and the diet after delete the wrong data'''
    uis = gconst.get_ui()
    # check the data
    data_ok = [] # check if the data is ok
    reasons = [] # the reason of the data is not ok
    diet_feature = gconst.get_labels_diet()
    for i in range(len(diet_dict[diet_feature[0]])):
        nutritions_ok = True
        reason = ''
        sum_nutrition = 0 # keep the sum of nutrition is not 0
        for feature in diet_feature:
            sum_nutrition += diet_dict[feature][i]
        if sum_nutrition <1:
            nutritions_ok = False
            reason += '0 '
        for ui in uis:
            if diet_dict[ui][i]>uis[ui]*times:
                nutritions_ok = False
                reason += f'{ui} '
        data_ok.append(nutritions_ok)
        reasons.append(reason)
    # get the index of the data is ok
    index_ok = [i for i in range(len(data_ok)) if data_ok[i]]
    diet_ok = {}
    for key in diet_dict:
        diet_ok[key] = []
    for index in index_ok:
        for key in diet_dict:
            diet_ok[key].append(diet_dict[key][index])
    return data_ok,reasons,diet_ok
def get_ok_diet(diet,times=1.5):
    '''input diet, return the diet after delete the wrong data'''
    diet_new = {}
    for id in diet:
        is_ok,reasons,diet_one = get_ok_diet_for_one(diet[id],times)
        diet_new[id] = diet_one
    return diet_new
def divide_diet(diet,times,result_path):
    '''divide the diet into ok and not ok, and save to csv file'''
    diet_dict = diet_dict2dict(diet)
    is_oks,reasons,ok_diets = get_ok_diet_for_one(diet_dict,times)
    diet_notok = {'reason':[]}
    diet_ok = {}
    for key in diet_dict:
        diet_notok[key] = []
        diet_ok[key] = []
    
    for index,is_ok in enumerate(is_oks):
        if is_ok:
            for key in diet_dict:
                diet_ok[key].append(diet_dict[key][index])
        else:
            diet_notok['reason'].append(reasons[index])
            for key in diet_dict:
                diet_notok[key].append(diet_dict[key][index])
    with open(result_path + 'diet_notok.csv','w') as f:
        pd.DataFrame(diet_notok).to_csv(f,index=False,header=True)
    with open(result_path + 'diet_ok.csv','w') as f:
        pd.DataFrame(diet_ok).to_csv(f,index=False,header=True)
def get_diet_in(diet,pdid_656):
    '''return the diet in pdid_656'''
    diet_new = {}
    pdid_656 = set(pdid_656)
    for id in diet:
        if id in pdid_656:
            diet_new[id] = diet[id]
    return diet_new
    
def norm_fit(matrix,labels,result_path,dump=False):
    '''calculate the scalers for normalization'''
    scalers = {}
    for i in range(matrix.shape[1]):
        scaler = preprocessing.StandardScaler()
        scalers[labels[i]] = scaler.fit(matrix[:,i].reshape(-1,1))
    if dump:
        with open(f'{result_path}scalers_diet','wb') as f:
            pickle.dump(scalers,f)
    return scalers
def norm_transform(scalers,matrix,labels,result_path='../data/healthier/',result_name='656_diet',dump=False):
    '''transform the matrix to normalized matrix'''
    x = copy.deepcopy(matrix)
    for i in range(x.shape[1]):
            x[:,i] = scalers[labels[i]].transform(x[:,i].reshape(-1,1)).squeeze()
    if dump:
        with open(f'{result_path}{result_name}','wb') as f:
            pickle.dump(x,f)
    return x
def get_diet_visit_align_index(pdid,diet,visit_dates):
    '''diet and visit align, for each diet, find the nearest visit index'''
    results = []
    for index,id in enumerate(pdid):
        cur_result = []
        cur_diet_dates = diet[id]['date']
        for cur_diet_date in cur_diet_dates:
            date1 = datetime.datetime.strptime(cur_diet_date,'%Y/%m/%d')
            sub = [abs((datetime.datetime.strptime(i,'%Y/%m/%d') - date1).days) for i in visit_dates[index]]
            min_index = sub.index(min(sub))
            cur_result.append(min_index)
        results.append(cur_result)
    return results
def main():
    logger.info(f'main: start.')
    args = parse_arg()
    data_path = args.data_path
    origin_data_path = args.origin_data_path
    result_path = args.result_path
    logger.info('load data: start.')

    pdid_656 = pickle.load(open(f'{data_path}656_pdid','rb'))
    x_656 = pickle.load(open(f'{data_path}656_x','rb'))
    y_656 = pickle.load(open(f'{data_path}656_y','rb'))
    static_656 = pickle.load(open(f'{data_path}656_static','rb'))


    pdid_669 = pickle.load(open(f'{data_path}669_pdid','rb'))
    x_669 = pickle.load(open(f'{data_path}raw_x_17_669','rb'))
    y_669 = pickle.load(open(f'{data_path}669_y','rb'))
    static_669 = pickle.load(open(f'{data_path}raw_static_4_669','rb'))

    diet = pickle.load(open(f'{origin_data_path}diet_merge','rb'))
    diet_656 = get_diet_in(diet,pdid_656)
    # data with date
    raw_x_669 = pickle.load(open(f'{data_path}raw_669.dict','rb'))
    date = pickle.load(open(f'{data_path}656_date','rb'))


    count = 0
    for i in range(656):
        if(len(x_656[i])!=len(y_656[i])):
            count += 1
    logger.debug(f'check the length of x and y in 656: {len(x_656)} {len(y_656)} , different count: {count}')
    logger.info('load data:done.')
    if args.task_output_diet:
        logger.info('task_output_diet: start.')
        output_diet(diet,result_path + 'diet.csv')
        logger.info('task_output_diet: end.')
    if args.task_pic_distribution:
        logger.info('task_pic_distribution: start.')
        diet_656 = get_diet_in(diet,pdid_656)
        pic_distribution(diet_656,result_path,f'diet_distribution.png',pairplot=True)
        logger.info('task_pic_distribution: end.')
    if args.task_pick_diet:
        logger.info('task_pick_diet: start.')
        diet_new = get_ok_diet(diet,args.pick_nutrition_times)

        logger.debug(f'patients number in origin_dataset: {len(list(diet.keys()))}')
        logger.debug(f'patients number in origin_dataset and in pdid_656: {len(list(set(diet.keys()).intersection(set(pdid_656))))}')
        count_656 = 0
        count_746 = 0
        for id in diet:
            if id in set(pdid_656):
                count_656+= len(diet[id]['date'])
            count_746+= len(diet[id]['date'])
        logger.debug(f'origin dataset: 656:{count_656} 746:{count_746}')
        count_656 = 0
        count_746 = 0
        for id in diet_new :
            if id in set(pdid_656):
                count_656+= len(diet_new[id]['date'])
            count_746+= len(diet_new[id]['date'])
        logger.debug(f'picked dataset: 656:{count_656} 746:{count_746}')
        # save
        with open(f'{data_path}diet656_picked.dict','wb') as f:
            pickle.dump(get_diet_in(diet_new,pdid_656),f) 

        logger.info('task_pick_diet: end.')
    if args.task_divide_diet:
        logger.debug('task_divide_diet: start.')
        divide_diet(get_diet_in(diet,pdid_656),args.pick_nutrition_times,result_path)
        logger.debug('task_divide_diet: end.')
    if args.task_norm_fit:
        logger.debug('task_norm_fit: start.')
        train_diet_dict = get_diet_part(diet,gconst.get_train_pdid(pdid_656))
        train_diet_matrix = diet_dict2matrix(train_diet_dict,no_id=True,no_date=True)
        scalers = norm_fit(train_diet_matrix,gconst.get_labels_diet(),data_path,dump=True)
        norm_transform(scalers,train_diet_matrix,gconst.get_labels_diet(),data_path,result_name='656_diet',dump=False)
        logger.debug('task_norm_fit: end.')
    if args.task_diet_visit_align:
        logger.debug('task_diet_visit_align: start.')
        get_diet_visit_align_index(pdid_656,diet_656,date)
        logger.debug('task_diet_visit_align: done.')

def parse_arg() ->argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/healthier/', help='Processed data path')
    parser.add_argument('--origin_data_path', type=str, default='./data/origin/ckd/', help='Original data path')
    parser.add_argument('--result_path', type=str, default='./result_sci/', help='Result path')
    parser.add_argument('--pick_nutrition_times', type=float, default=1.5, help='Help to remove unreasonable values ')

    # task
    parser.add_argument('--task_pick_diet', default=False, action='store_true', help='Pick out possibly incorrect data in diet656 and save it to binary files')
    parser.add_argument('--task_divide_diet', default=False, action='store_true', help='Separate correct data and incorrect data in diet656 and output to csv files')
    parser.add_argument('--task_norm_fit', default=False, action='store_true', help='Normalize the fit part and save scalers to data_path')
    parser.add_argument('--task_output_diet', default=False, action='store_true', help='Output origin diet to csv files')
    parser.add_argument('--task_pic_distribution', default=False, action='store_true', help='Plot the distribution of each indicator in diet656 and save images')
    parser.add_argument('--task_diet_visit_align', default=False, action='store_true', help='Alignment between diet and visit')


    return parser.parse_args()

if __name__ == '__main__':
    print('hello')
    main()