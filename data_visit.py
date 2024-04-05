'''
process the visit data
'''
from cmath import nan
from email.policy import default
import math
from operator import index
from unittest import result
from loguru import logger
import numpy as np
import argparse
import os
import pickle
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import io
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import copy
from dateutil.relativedelta import relativedelta


import global_const as gconst
def get_date_part(date,indexs):
    return [date[i] for i in index]
def pick_visit():
    args = parse_args()
    # load data
    logger.info('loading data..')
    origin_data_path = args.origin_data_path
    data_path = args.data_path # './data/healthier/'

    survival_dict = pickle.load(
        open(origin_data_path+'survival', 'rb'))
    blood_dict = pickle.load(
        open(origin_data_path+'blood', 'rb'))
    body_dict = pickle.load(
        open(origin_data_path+'body_dict', 'rb'))
    dialysis_dict = pickle.load(open(
        origin_data_path+'dialysis_sufficiency.dict', 'rb'))
    diet_dict = pickle.load(
        open(origin_data_path+'diet_merge', 'rb'))
    disease_dict = pickle.load(
        open(origin_data_path+'disease', 'rb'))
    vital_sign = pickle.load(open(origin_data_path+'vital_sign','rb'))

    x_656 = pickle.load(open(origin_data_path + 'x_656', 'rb'))
    pdid_656 = pickle.load(open(origin_data_path + 'pdid_656', 'rb'))

    # 同时存在于这里的人的pdid：survival, blood, diet_dict, body_list(有身高数据), vital_sign(日期、收缩压、舒张压、心跳)
    logger.info('get intersection from: survival, blood, diet_dict, body_list( have height), vital_sign(日期、收缩压、舒张压、心跳)')
    blood_list = list()
    for pdid in blood_dict:
        if pdid in survival_dict:
            blood_list.append(pdid)
    blood_count = len(blood_list)
    survival_count = len(survival_dict)
    body_count = 0
    body_list = list()
    diet_count = len(diet_dict)
    for pdid in survival_dict:
        flag = 0
        if str(survival_dict[pdid]['Height']) != 'nan':
            flag = 1
        if pdid in body_dict:
            for i in range(len(body_dict[pdid]['height'])):
                if str(body_dict[pdid]['height'][i]) != 'nan':
                    flag = 1
                    break
        if pdid in dialysis_dict:
            for i in range(len(dialysis_dict[pdid]['身高'])):
                if str(dialysis_dict[pdid]['身高'][i]) != 'nan':
                    flag = 1
                    break
        if flag == 1:
            body_list.append(pdid)
    body_count = len(body_list)
    final_pdid = list()
    for pdid in survival_dict:
        if pdid in blood_dict and pdid in diet_dict and pdid in body_list and pdid in vital_sign:
            final_pdid.append(pdid)
    final_count = len(final_pdid)
    print('Survival: %d Blood: %d Body: %d Final: %d ' %
        (survival_count, blood_count, body_count, final_count))

    # Get observation windows end date 获取死亡时间的最后一天
    logger.info('get observation windows end date')
    last_death = datetime.datetime.strptime('2000/01/01', '%Y/%m/%d')
    for pdid in survival_dict:
        if survival_dict[pdid]['Death'] == 1:
            last_date = survival_dict[pdid]['Death_date']
            if str(last_date) != 'nan':
                last_date = datetime.datetime.strptime(last_date, '%Y/%m/%d')
                if last_date > last_death:
                    last_death = last_date
    print('Observation window ended at: %s' % last_death)


    # 对于存活的人，找到上面last_death一年前的index；对于死亡的人，找到他自己死亡一年前的index
    # 此处会减掉53人的1-2条数据，所以x_656/x_662中的某些患者的数据会比blood少1-2条
    logger.info("get data's labels by cutting some visits")
    raw_dataset = dict()
    for pdid in final_pdid:
        if int(pdid) not in survival_dict.keys():
            continue
        end_index = len(blood_dict[pdid]['date'])
        if survival_dict[pdid]['Death'] == 0:
            for i in range(len(blood_dict[pdid]['date'])):
                cur_date = datetime.datetime.strptime(
                    blood_dict[pdid]['date'][i], '%Y/%m/%d')
                if (last_death-cur_date).days < 365:
                    end_index = i
                    break
        if survival_dict[pdid]['Death'] == 1 and int(survival_dict[pdid]['Death_number']/10) == 9:
            survival_dict[pdid]['Death'] = 0
            survival_dict[pdid]['Death_date'] = np.nan
            for i in range(len(blood_dict[pdid]['date'])):
                cur_date = datetime.datetime.strptime(
                    blood_dict[pdid]['date'][i], '%Y/%m/%d')
                if (last_death-cur_date).days < 365:
                    end_index = i
                    break
        if end_index == 0:
            continue

        raw_dataset[int(pdid)] = dict()
        raw_dataset[int(pdid)]['name'] = blood_dict[pdid]['name']
        raw_dataset[int(pdid)]['date'] = blood_dict[pdid]['date'][:end_index]
        #feature_list = ['血氯', '血二氧化碳结合力', '血尿酸', '血白细胞计数', '血红蛋白', 'urea', 'ca', 'k', '血钠', 'scr', 'p', 'albumin', '血甲状旁腺激素', '血超敏C反应蛋白', '血甘油三酯', '血清铁', '血糖', '血总胆固醇', '血高密度脂蛋白', '血低密度脂蛋白']
        feature_list = ['血氯', '血二氧化碳结合力', '血白细胞计数', '血红蛋白', 'urea', 'ca', 'k', '血钠', 'scr', 'p', 'albumin', '血超敏C反应蛋白', '血糖']
        for each_feature in feature_list:
            raw_dataset[int(pdid)][each_feature] = blood_dict[pdid][each_feature][:end_index]

    
    # 看此时的对应id的患者的的x_656数据条数是否与删减后的raw_dataset 相等，结论是相等，可见x_656等的原始数据即是现在的raw_data
    logger.info('check : compare x_656 with raw_data to find difference between them')
    
    count = 0
    for i in range(656):
        if len(x_656[i]) != len(raw_dataset[pdid_656[i]]['date']):
            count += 1
    print(f'has difference : {count}')


    # 找出整体缺失率3/5以上 (0.95*13/8=0.58) 某项指标的缺失不超过3/5
    logger.info('find pdid who lost more than 3/5 data in one feature')
    delete_list = list()
    del_count = 0
    for pdid in raw_dataset:
        count = 0
        for f in feature_list:
            count += sum(np.isnan(raw_dataset[pdid][f])) # 某位患者某项指标的缺失个数
            if sum(np.isnan(raw_dataset[pdid][f])) == len(raw_dataset[pdid]['date']): # 如果有一项指标完全缺失，那么直接删除
                del_count += 1
                delete_list.append(pdid)
                break
        if count / (8*len(raw_dataset[pdid]['date'])) >= 0.95 and pdid not in delete_list: # 缺失率过高的患者，我们删除他/她
            del_count += 1
            delete_list.append(pdid)
    print('Delete: %d Remain: %d' % (del_count, len(raw_dataset)-del_count))
    for pdid in delete_list:
        del raw_dataset[pdid]



    # 补全, 策略是找最近一个不空的日期，用其代替
    logger.info('fill the data by using the closest data')
    for pdid in raw_dataset:
        for f in feature_list:
            for i in range(len(raw_dataset[pdid]['date'])):
                if str(raw_dataset[pdid][f][i]) == 'nan': # 患者，指标，每天 如果是nan，需要补全
                    if i == 0: # 第一天
                        val = 0
                        for j in range(len(raw_dataset[pdid]['date'])):
                            if str(raw_dataset[pdid][f][j]) != 'nan':
                                val = raw_dataset[pdid][f][j]
                                break
                        raw_dataset[pdid][f][i] = val
                    elif i == len(raw_dataset[pdid]['date'])-1:
                        val = 0
                        for j in range(len(raw_dataset[pdid]['date'])-1, -1, -1):
                            if str(raw_dataset[pdid][f][j]) != 'nan':
                                val = raw_dataset[pdid][f][j]
                                break
                        raw_dataset[pdid][f][i] = val
                    else:
                        cur_date = raw_dataset[pdid]['date'][i]
                        prev_val = 0
                        prev_date = 0
                        post_val = 0
                        post_date = 0
                        
                        for j in range(i-1, -1, -1): # 往前找最近一个有指标的日期，记录日期和指标
                            if str(raw_dataset[pdid][f][j]) != 'nan':
                                prev_val = raw_dataset[pdid][f][j]
                                prev_date = raw_dataset[pdid]['date'][j]
                                break
                        for j in range(i+1, len(raw_dataset[pdid]['date'])): # 往后找最近一个有指标的日期，记录日期和指标
                            if str(raw_dataset[pdid][f][j]) != 'nan':
                                post_val = raw_dataset[pdid][f][j]
                                post_date = raw_dataset[pdid]['date'][j]
                                break
                        cur_date = datetime.datetime.strptime(cur_date, '%Y/%m/%d')
                        prev_date = datetime.datetime.strptime(
                            prev_date, '%Y/%m/%d')
                        try:
                            post_date = datetime.datetime.strptime(
                                post_date, '%Y/%m/%d')
                        except:
                            pass
                        if post_date == 0 or prev_date - cur_date <= post_date - cur_date: # 没找到之后的天，或和前面一天距离短，用前面一天的替代
                            raw_dataset[pdid][f][i] = prev_val
                        else: # 否则用后面一天的替代
                            raw_dataset[pdid][f][i] = post_val


    # 筛选死亡时间有误的患者
    logger.info('find pdid has wrong death date')
    pdid_list = list()
    pdid_length = list()
    dataset = list()
    death_time = list()
    live_count = 0
    death_count = 0
    count = 0
    err_count = 0
    for pdid in raw_dataset:
        last_date = blood_dict[pdid]['date'][-1]
        last_date = datetime.datetime.strptime(last_date, '%Y/%m/%d')
        first_date = blood_dict[pdid]['date'][0]
        first_date = datetime.datetime.strptime(first_date, '%Y/%m/%d')
        pdid_list.append(pdid)
        pdid_length.append(len(raw_dataset[pdid]['date']))
        for i in range(len(feature_list)): # 一行是一个指标（有date个），堆叠起来
            if i == 0:
                arr = np.array(raw_dataset[pdid][feature_list[i]])
            else:
                arr = np.vstack((arr, raw_dataset[pdid][feature_list[i]]))
        dataset.append(arr.transpose()) # 转置，变成一行是一天的，每天有13条指标
        
        death_date = survival_dict[pdid]['Death_date']
        if str(death_date) == 'nan':
            death_time.append(-1)
            live_count += 1
            continue
        death_date = datetime.datetime.strptime(death_date, '%Y/%m/%d')
        delt = (death_date-last_date).days
        # 死亡时间有错
        if delt < 0: # 死亡时间比最后一天还往后
            err_count += 1
            death_time.append(1)
            death_count += 1
        elif delt > 365 or int(survival_dict[pdid]['Death_number']/10) == 9: # 最后一次记录是死亡时间一年前，说明可能转院或回家或放弃治疗一类的，说明一年后还可存活
            death_time.append(-1)
            count += 1
            live_count += 1
        else: # 死亡时间没错
            death_time.append(delt)
            death_count += 1
    print('Fixed %d error death time.' % err_count)
    print('%d patients died over 365 days.' % count)
    print('Death: %d, Live: %d' % (death_count, live_count))

    # 为每个人的每个有指标记录的天，找最近的一个饮食记录
    logger.info('find the closest diet record for each visit')
    diet_aligned = dict()
    diet_feature = ['water', 'protein', 'fat', 'carbohydrate', 'df', 'Calories']
    for pdid in raw_dataset:
        diet_aligned[pdid] = dict()
        diet_aligned[pdid]['date'] = list()
        for each_feature in diet_feature:
            diet_aligned[pdid][each_feature] = list()
    for pdid in raw_dataset:
        for i in range(len(raw_dataset[pdid]['date'])):
            diet_aligned[pdid]['date'].append(raw_dataset[pdid]['date'][i])
            target_date = datetime.datetime.strptime(
                raw_dataset[pdid]['date'][i], '%Y/%m/%d')
            # Search the most recent diet record
            ind = -1
            min_delt = 9999
            for j in range(len(diet_dict[pdid]['date'])):
                cur_date = datetime.datetime.strptime(
                    diet_dict[pdid]['date'][j], '%Y/%m/%d')
                cur_delt = abs((target_date-cur_date).days)
                if cur_delt < min_delt:
                    min_delt = cur_delt
                    ind = j
            for each_feature in diet_feature:
                diet_aligned[pdid][each_feature].append(
                    diet_dict[pdid][each_feature][ind])
    
    # 统计吃的食物的量
    logger.info('calculate the appetite')
    for pdid in diet_aligned:
        diet_aligned[pdid]['amount'] = list()
        for i in range(len(diet_aligned[pdid]['date'])):
            amount = 0
            for each_feature in diet_feature:
                amount += diet_aligned[pdid][each_feature][i]
            diet_aligned[pdid]['amount'].append(amount)
    # 查找身高和体重数据
    logger.info('find height and weight for each pdid')
    body_aligned = dict()
    for pdid in raw_dataset:
        height_dict = {'date': [], 'height': []}
        weight_dict = {'date': [], 'weight': []}

        # Get all height
        if pdid in body_dict:
            for i in range(len(body_dict[pdid]['date'])):
                if str(body_dict[pdid]['height'][i]) != 'nan':
                    height_dict['date'].append(body_dict[pdid]['date'][i])
                    height_dict['height'].append(body_dict[pdid]['height'][i])
        if pdid in dialysis_dict:
            for i in range(len(dialysis_dict[pdid]['日期'])):
                if str(dialysis_dict[pdid]['身高'][i]) != 'nan':
                    cur_date = dialysis_dict[pdid]['日期'][i].to_pydatetime()
                    date_str = '%d/%d/%d' % (cur_date.year,
                                            cur_date.month, cur_date.day)
                    height_dict['date'].append(date_str)
                    height_dict['height'].append(dialysis_dict[pdid]['身高'][i])
        if len(height_dict['date']) == 0:
            height_dict['date'].append('2000/1/1')
            height_dict['height'].append(survival_dict[pdid]['Height'])
        assert(len(height_dict['date']) != 0)

        # Get all weight
        if pdid in body_dict:
            for i in range(len(body_dict[pdid]['date'])):
                if str(body_dict[pdid]['weight'][i]) != 'nan':
                    weight_dict['date'].append(body_dict[pdid]['date'][i])
                    weight_dict['weight'].append(body_dict[pdid]['weight'][i])
        if pdid in dialysis_dict:
            for i in range(len(dialysis_dict[pdid]['日期'])):
                if str(dialysis_dict[pdid]['实际体重'][i]) != 'nan':
                    cur_date = dialysis_dict[pdid]['日期'][i].to_pydatetime()
                    date_str = '%d/%d/%d' % (cur_date.year,
                                            cur_date.month, cur_date.day)
                    weight_dict['date'].append(date_str)
                    weight_dict['weight'].append(dialysis_dict[pdid]['实际体重'][i])
        if len(weight_dict['date']) == 0:
            weight_dict['date'].append('2000/1/1')
            weight_dict['weight'].append(survival_dict[pdid]['Weight'])
        assert(len(weight_dict['date']) != 0)

        # Sort
        for i in range(len(height_dict['date'])):
            for j in range(len(height_dict['date'])-1):
                if datetime.datetime.strptime(height_dict['date'][j], '%Y/%m/%d') > datetime.datetime.strptime(height_dict['date'][j+1], '%Y/%m/%d'):
                    height_dict['date'][j], height_dict['date'][j +
                                                                1] = height_dict['date'][j+1], height_dict['date'][j]
                    height_dict['height'][j], height_dict['height'][j +
                                                                    1] = height_dict['height'][j+1], height_dict['height'][j]
        for i in range(len(weight_dict['date'])):
            for j in range(len(weight_dict['date'])-1):
                if datetime.datetime.strptime(weight_dict['date'][j], '%Y/%m/%d') > datetime.datetime.strptime(weight_dict['date'][j+1], '%Y/%m/%d'):
                    weight_dict['date'][j], weight_dict['date'][j +
                                                                1] = weight_dict['date'][j+1], weight_dict['date'][j]
                    weight_dict['weight'][j], weight_dict['weight'][j +
                                                                    1] = weight_dict['weight'][j+1], weight_dict['weight'][j]

        body_aligned[pdid] = {'date': [], 'height': [], 'weight': []}
        for i in range(len(raw_dataset[pdid]['date'])):
            body_aligned[pdid]['date'].append(raw_dataset[pdid]['date'][i])
            target_date = datetime.datetime.strptime(
                raw_dataset[pdid]['date'][i], '%Y/%m/%d')

            # Height
            # find cur index
            val = np.nan
            cur_ind = len(height_dict['date'])
            for j in range(len(height_dict['date'])):
                if datetime.datetime.strptime(height_dict['date'][j], '%Y/%m/%d') > target_date:
                    cur_ind = j
                    break
            # Search pre record
            pre_ind = cur_ind-1
            if pre_ind >= 0:
                pre_delt = abs(
                    (target_date-datetime.datetime.strptime(height_dict['date'][pre_ind], '%Y/%m/%d')).days)
            post_ind = cur_ind+1
            if post_ind < len(height_dict['date']):
                post_delt = abs(
                    (target_date-datetime.datetime.strptime(height_dict['date'][post_ind], '%Y/%m/%d')).days)
            if pre_ind >= 0 and post_ind < len(height_dict['date']):
                total_change = height_dict['height'][post_ind] - \
                    height_dict['height'][pre_ind]
                if pre_delt < post_delt:
                    val = height_dict['height'][pre_ind] + \
                        pre_delt * total_change/(pre_delt+post_delt)
                else:
                    val = height_dict['height'][post_ind] - \
                        post_delt * total_change/(pre_delt+post_delt)
            elif pre_ind < 0:
                val = height_dict['height'][0]
            elif post_ind >= len(height_dict['date']):
                val = height_dict['height'][-1]
            assert(str(val) != 'nan')
            body_aligned[pdid]['height'].append(val)

            # Weight
            # find cur index
            val = np.nan
            cur_ind = len(weight_dict['date'])
            for j in range(len(weight_dict['date'])):
                if datetime.datetime.strptime(weight_dict['date'][j], '%Y/%m/%d') > target_date:
                    cur_ind = j
                    break
            # Search pre record
            pre_ind = cur_ind-1
            if pre_ind >= 0:
                pre_delt = abs(
                    (target_date-datetime.datetime.strptime(weight_dict['date'][pre_ind], '%Y/%m/%d')).days)
            post_ind = cur_ind+1
            if post_ind < len(weight_dict['date']):
                post_delt = abs(
                    (target_date-datetime.datetime.strptime(weight_dict['date'][post_ind], '%Y/%m/%d')).days)
            if pre_ind >= 0 and post_ind < len(weight_dict['date']):
                total_change = weight_dict['weight'][post_ind] - \
                    weight_dict['weight'][pre_ind]
                if pre_delt < post_delt:
                    val = weight_dict['weight'][pre_ind] + \
                        pre_delt * total_change/(pre_delt+post_delt)
                else:
                    val = weight_dict['weight'][post_ind] - \
                        post_delt * total_change/(pre_delt+post_delt)
            elif pre_ind < 0:
                val = weight_dict['weight'][0]
            elif post_ind >= len(weight_dict['date']):
                val = weight_dict['weight'][-1]
            if str(val) == 'nan':
                min_ind = -1
                min_delt = 99999
                for j in range(len(height_dict['date'])):
                    cur_date = datetime.datetime.strptime(
                        height_dict['date'][j], '%Y/%m/%d')
                    cur_delt = abs((target_date-cur_date).days)
                    if cur_delt < min_delt:
                        min_delt = cur_delt
                        min_ind = j
                val = height_dict['height'][min_ind]-105
            assert(str(val) != 'nan')
            body_aligned[pdid]['weight'].append(val)

    # 查找血压数据，高压低压，找最近的一天替代缺失值
    logger.info('find systolic and diastolic for each patient')
    vital_aligned = dict()
    vital_feature = ['systolic', 'diastolic']
    for pdid in raw_dataset:
        vital_aligned[pdid] = dict()
        vital_aligned[pdid]['date'] = list()
        for each_feature in vital_feature:
            vital_aligned[pdid][each_feature] = list()
    for pdid in raw_dataset:
        for i in range(len(raw_dataset[pdid]['date'])):
            vital_aligned[pdid]['date'].append(raw_dataset[pdid]['date'][i])
            target_date = datetime.datetime.strptime(
                raw_dataset[pdid]['date'][i], '%Y/%m/%d')
            # Search the most recent diet record
            ind = -1
            min_delt = 9999
            for j in range(len(vital_sign[pdid]['date'])):
                cur_date = datetime.datetime.strptime(vital_sign[pdid]['date'][j], '%Y/%m/%d')
                cur_delt = abs((target_date-cur_date).days)
                if cur_delt < min_delt and not np.isnan(vital_sign[pdid]['systolic'][j]):
                    min_delt = cur_delt
                    ind = j
            vital_aligned[pdid]['systolic'].append(vital_sign[pdid]['systolic'][ind])
                
            ind = -1
            min_delt = 9999
            for j in range(len(vital_sign[pdid]['date'])):
                cur_date = datetime.datetime.strptime(vital_sign[pdid]['date'][j], '%Y/%m/%d')
                cur_delt = abs((target_date-cur_date).days)
                if cur_delt < min_delt and not np.isnan(vital_sign[pdid]['diastolic'][j]):
                    min_delt = cur_delt
                    ind = j
            vital_aligned[pdid]['diastolic'].append(vital_sign[pdid]['diastolic'][ind])

    # 查找static数据
    logger.info('static data')
    age_list = list()
    gender_list = list()
    diab_list = list()
    height_list = list()
    for i in range(len(pdid_list)):
        if str(survival_dict[pdid_list[i]]['Age']) == 'nan':
            age_list.append(59.23)
        else:
            age_list.append(survival_dict[pdid_list[i]]['Age'])
        if str(survival_dict[pdid_list[i]]['Gender']) == 'nan':
            gender_list.append(0)
        else:
            gender_list.append(survival_dict[pdid_list[i]]['Gender'])
        if str(survival_dict[pdid_list[i]]['Diabetes']) == 'nan':
            diab_list.append(0)
        else:
            diab_list.append(survival_dict[pdid_list[i]]['Diabetes'])
        height_list.append(body_aligned[pdid_list[i]]['height'][0])
    age_arr = sklearn.preprocessing.StandardScaler().fit_transform(np.array(age_list).reshape(-1, 1))
    height_arr = sklearn.preprocessing.StandardScaler().fit_transform(np.array(height_list).reshape(-1, 1))
    gender_arr = np.array(gender_list).reshape(-1, 1)
    diab_arr = np.array(diab_list).reshape(-1, 1)
    static_dataset = list(np.hstack([age_arr, gender_arr, diab_arr, height_arr]))
    print(static_dataset[0])

    # 做差的time
    logger.info('time sub')
    time_dataset = list()
    for pdid in raw_dataset:
        cur_list = [0]
        for i in range(1,len(raw_dataset[pdid]['date'])):
            prev_date = datetime.datetime.strptime(raw_dataset[pdid]['date'][i-1], '%Y/%m/%d')
            cur_date = datetime.datetime.strptime(raw_dataset[pdid]['date'][i], '%Y/%m/%d')
            cur_list.append((cur_date-prev_date).days / 30)
            assert(cur_list[-1]>=0)
        time_dataset.append(cur_list)
        
    for i in range(len(time_dataset)):
        for j in range(1,len(time_dataset[i])):
            time_dataset[i][j] = time_dataset[i][j] + time_dataset[i][j-1]

    # 做差的time,expend
    # 目的是要知道当前时间和头一天的差值
    time_matrix = list()
    for i in range(len(time_dataset)):
        cur_matrix = list()
        for j in range(len(time_dataset[i])):
            tmp = time_dataset[i][:j+1]
            tmp.reverse()
            cur_matrix.append(tmp)
        for j in range(len(cur_matrix)):
            for k in range(len(cur_matrix[j]),len(time_dataset[i])):
                cur_matrix[j].append(0)
        cur_matrix = np.array(cur_matrix)
        time_matrix.append(cur_matrix) 



    # 合并13个血液检测特征和4个体测特征  ['Cl', 'CO2CP', 'WBC', 'Hb', 'Urea', 'Ca', 'K', 'Na', 'Scr', 'P', 'Albumin', 'hs-CRP', 'Glucose']+ ['Appetite','Weight','SBP','DBP']
    logger.info('merge dataset : 13 + 4 = 17')
    merge_dataset = copy.deepcopy(dataset)
    for i in range(len(pdid_list)):
        assert(len(diet_aligned[pdid_list[i]]['amount']) == dataset[i].shape[0])
        assert(len(body_aligned[pdid_list[i]]['weight']) == dataset[i].shape[0])
        assert(len(vital_aligned[pdid_list[i]]['systolic']) == dataset[i].shape[0])
        assert(len(vital_aligned[pdid_list[i]]['diastolic']) == dataset[i].shape[0])

        amout_arr = np.array(diet_aligned[pdid_list[i]]['amount']).reshape(
            dataset[i].shape[0], 1)
        weight_arr = np.array(body_aligned[pdid_list[i]]['weight']).reshape(
            dataset[i].shape[0], 1)
        sys_arr = np.array(vital_aligned[pdid_list[i]]['systolic']).reshape(
            dataset[i].shape[0], 1)
        dias_arr = np.array(vital_aligned[pdid_list[i]]['diastolic']).reshape(
            dataset[i].shape[0], 1)
        merge_arr = np.hstack([amout_arr, weight_arr, sys_arr, dias_arr])

        merge_dataset[i] = np.hstack([merge_dataset[i], merge_arr])
    print(merge_dataset[0].shape)
    print(len(merge_dataset))


    # dict转化为矩阵并归一化
    logger.info('normalization for visit')
    transform_data = np.array([])
    for i in range(len(merge_dataset)):
        val = merge_dataset[i]
        if len(transform_data) == 0:
            transform_data = val
        else:
            transform_data = np.vstack((transform_data, val))
    print(transform_data.shape)

    scalers = {}
    for i in range(transform_data.shape[1]):
        scalers[i] = sklearn.preprocessing.StandardScaler()
        scalers[i] = scalers[i].fit(transform_data[:, i].reshape(-1, 1))
    x = copy.deepcopy(merge_dataset)
    for j in range(len(x)):
        for i in range(x[j].shape[1]):
            x[j][:, i] = scalers[i].transform(x[j][:, i].reshape(-1, 1)).squeeze()
            

    # 时间归一化
    logger.info('normalization for time')
    time_scaler = sklearn.preprocessing.StandardScaler()

    val = list()
    for i in range(len(time_matrix)):
        for j in range(len(time_matrix[i])):
            val+=list(time_matrix[i][j])        
    time_scaler = time_scaler.fit(np.array(val).reshape(-1,1))
    for i in range(len(time_matrix)):
        for j in range(len(time_matrix[i])):
            time_matrix[i][j] = time_scaler.transform(time_matrix[i][j].reshape(-1, 1)).reshape(1, -1)


    # 计算y
    logger.info('calculate y')
    y_n2n = list()
    sample_weight = list()
    emergency_reason = []
    for pdid in survival_dict:
        if str(survival_dict[pdid]['Death_number']) != 'nan':
            if survival_dict[pdid]['Death_number'] not in emergency_reason:
                emergency_reason.append(survival_dict[pdid]['Death_number'])

    for i in range(len(pdid_list)):
        pdid = pdid_list[i]
        cur_label = np.zeros(merge_dataset[i].shape[0])
        cur_weight = np.ones(merge_dataset[i].shape[0])
        # Survival
        if death_time[i] == -1:
            y_n2n.append(cur_label)
            sample_weight.append(cur_weight)
            continue

        death_date = survival_dict[pdid]['Death_date']
        death_date = datetime.datetime.strptime(death_date, '%Y/%m/%d')
        for j in range(len(blood_dict[pdid]['date'])):
            cur_date = blood_dict[pdid]['date'][j]
            cur_date = datetime.datetime.strptime(cur_date, '%Y/%m/%d')
            if (death_date-cur_date).days > 365*2:
                cur_weight[j] = 1
            elif (death_date-cur_date).days > 365:
                cur_weight[j] = 0
            elif survival_dict[pdid]['Death_number'] in emergency_reason and (death_date-cur_date).days > 180:
                cur_label[j] = 1
                cur_weight[j] = 0
            else:
                cur_label[j] = 1
                cur_weight[j] = 2
        y_n2n.append(cur_label)
        sample_weight.append(cur_weight)
        assert(cur_label[-1] == 1)
    assert(len(y_n2n) == len(x))
    assert(len(sample_weight) == len(x))


    total_count = 0
    survival_count = 0
    mask_count = 0
    death_count = 0
    for val in sample_weight:
        if sum(val) == 0:
            continue
        for w in val:
            if w == 1:
                survival_count += 1
            if w == 0:
                mask_count += 1
            if w == 2:
                death_count += 1
            total_count+=1
    print('Death: %d\nSurvive: %d\nMask: %d\nTotal:%d\nDeath ratio: %.2f%%\nMask ratio: %.2f%%\n' % (death_count,
                                                                    survival_count, mask_count,total_count, 100*death_count/survival_count, 100*mask_count/total_count))
    death_ratio = death_count/(total_count-death_count)



    logger.info('get the final dataset')
    for i in range(len(sample_weight)):
        for j in range(len(sample_weight[i])):
            if sample_weight[i][j] == 2:
                sample_weight[i][j] = 1            


    x2 = []
    y2 = []
    pdid_list2 = []
    time_matrix2 = []
    sample_weight2 = []
    static_dataset2 = []
    indexs = []

    for i in range(len(x)):
        if sum(sample_weight[i]) == 0:
            continue
        else:
            indexs.append(i)
            x2.append(x[i])
            y2.append(y_n2n[i])
            pdid_list2.append(pdid_list[i])
            time_matrix2.append(time_matrix[i])
            sample_weight2.append(sample_weight[i])
            static_dataset2.append(static_dataset[i])

    # 做差的time，单位天
    time_dataset_sub = list()
    for pdid in raw_dataset:
        cur_list = [0]
        for i in range(1,len(raw_dataset[pdid]['date'])):
            prev_date = datetime.datetime.strptime(raw_dataset[pdid]['date'][i-1], '%Y/%m/%d')
            cur_date = datetime.datetime.strptime(raw_dataset[pdid]['date'][i], '%Y/%m/%d')
            cur_list.append((cur_date-prev_date).days)
            assert(cur_list[-1]>=0)
        time_dataset_sub.append(cur_list)
    
    
    logger.info('saving data...')
    logger.info('save scalers')
    # 保存归一化 scalers: dynamic
    labels = ['Cl', 'CO2CP', 'WBC', 'Hb', 'Urea', 'Ca', 'K', 'Na', 'Scr', 'P', 'Albumin', 'hs-CRP', 'Glucose']+ ['Appetite','Weight','SBP','DBP']
    dump_scalers = {}
    for i in range(len(scalers)):
        dump_scalers[labels[i]] = scalers[i]

    with open(data_path + 'scalers_dynamic','wb') as f:
        pickle.dump(dump_scalers,f)
    # 保存归一化 scalers: dynamic+static
    labels = ['Cl', 'CO2CP', 'WBC', 'Hb', 'Urea', 'Ca', 'K', 'Na', 'Scr', 'P', 'Albumin', 'hs-CRP', 'Glucose']+ ['Appetite','Weight','SBP','DBP']
    dump_scalers = {}
    for i in range(len(scalers)):
        dump_scalers[labels[i]] = scalers[i]
    dump_scalers['age'] = sklearn.preprocessing.StandardScaler().fit(np.array(age_list).reshape(-1, 1))
    dump_scalers['height'] = sklearn.preprocessing.StandardScaler().fit(np.array(height_list).reshape(-1, 1))

    with open(data_path + 'scalers_dynamic_static','wb') as f:
        pickle.dump(dump_scalers,f)



    logger.info('save normalized data 662,669,656')
    # 处理好的归一化数据 662
    data_len = len(x2)
    pdid_662 = pdid_list2
    pickle.dump(x2, open(f"{data_path}{data_len}_x", 'wb')) # data_path/662_x
    pickle.dump(y2, open(f"{data_path}{data_len}_y", 'wb'))
    pickle.dump(pdid_list2, open(f"{data_path}{data_len}_pdid", 'wb'))
    pickle.dump(time_matrix2, open(f"{data_path}{data_len}_time", 'wb'))
    pickle.dump(sample_weight2, open(f"{data_path}{data_len}_weight", 'wb'))
    pickle.dump(static_dataset2, open(f"{data_path}{data_len}_static", 'wb'))

    # 处理好的归一化数据 669
    # 669
    data_len = len(x)
    pickle.dump(x, open(f"{data_path}{data_len}_x", 'wb'))
    pickle.dump(y_n2n, open(f"{data_path}{data_len}_y", 'wb'))
    pickle.dump(pdid_list, open(f"{data_path}{data_len}_pdid", 'wb'))
    pickle.dump(time_matrix, open(f"{data_path}{data_len}_time", 'wb'))
    pickle.dump(sample_weight, open(f"{data_path}{data_len}_weight", 'wb'))
    pickle.dump(static_dataset, open(f"{data_path}{data_len}_static", 'wb'))
    # print(len(x),len(y_n2n),len(pdid_list),len(time_matrix),len(sample_weight),len(static_dataset)) # 669

    # 处理好的归一化数据 656
    # 656
    pdid_669 = pdid_list
    data_len = len([x[pdid_669.index(id)] for id in pdid_656])
    pickle.dump([x[pdid_669.index(id)] for id in pdid_656], open(f"{data_path}{data_len}_x", 'wb'))
    pickle.dump([y_n2n[pdid_669.index(id)] for id in pdid_656], open(f"{data_path}{data_len}_y", 'wb'))
    pickle.dump([pdid_list[pdid_669.index(id)] for id in pdid_656], open(f"{data_path}{data_len}_pdid", 'wb'))
    pickle.dump([time_matrix[pdid_669.index(id)] for id in pdid_656], open(f"{data_path}{data_len}_time", 'wb'))
    pickle.dump([sample_weight[pdid_669.index(id)] for id in pdid_656], open(f"{data_path}{data_len}_weight", 'wb'))
    pickle.dump([static_dataset[pdid_669.index(id)] for id in pdid_656], open(f"{data_path}{data_len}_static", 'wb'))
    # print(len([x[pdid_669.index(id)] for id in pdid_656])) # 656


    # 保存未归一化的时间差
    logger.info('save date_sub (unnormalized)')
    data_len = len(time_dataset_sub)
    pickle.dump(time_dataset_sub,open(f'{data_path}{data_len}_date_sub','wb'))
    

    logger.info('save origin visit data (unnormalized)')
    # 662
    # 保存原始数据 与x2对应的 merge,filling 过的
    x_left = [merge_dataset[i] for i in indexs]
    pickle.dump(x_left,open(data_path + 'raw_x_17_662','wb'))

    # 656
    # 保存原始数据 与x_656对应的 merge,filling 过的

    pdid_669 = pdid_list
    x_left_index = [pdid_669.index(id) for id in pdid_656]
    x_left = [merge_dataset[i] for i in x_left_index]
    pickle.dump(x_left,open(data_path + 'raw_x_17_656','wb'))

    # 669
    pdid_669 = pdid_list
    x_left_index = [pdid_669.index(id) for id in pdid_669]
    x_left = [merge_dataset[i] for i in x_left_index]
    pickle.dump(x_left,open(data_path + 'raw_x_17_669','wb'))


    # 669 全部原始数据
    pickle.dump(raw_dataset,open(data_path + 'raw_669.dict','wb'))

    # 保存时间
    date_656 = [raw_dataset[id]['date'] for id in pdid_656]
    pickle.dump(date_656,open(data_path + '656_date','wb'))
    date_662 = [raw_dataset[id]['date'] for id in pdid_662]
    pickle.dump(date_662,open(data_path + '662_date','wb'))
    date_669 = [raw_dataset[id]['date'] for id in pdid_669]
    pickle.dump(date_669,open(data_path + '669_date','wb'))


    # 保存原始数据static，与归一化的对应
    # static 
    logger.info('save origin static data (unnormalized)')
    age_list = list()
    gender_list = list()
    diab_list = list()
    height_list = list()
    for i in range(len(pdid_list)):
        if str(survival_dict[pdid_list[i]]['Age']) == 'nan':
            age_list.append(59.23)
        else:
            age_list.append(survival_dict[pdid_list[i]]['Age'])
        if str(survival_dict[pdid_list[i]]['Gender']) == 'nan':
            gender_list.append(0)
        else:
            gender_list.append(survival_dict[pdid_list[i]]['Gender'])
        if str(survival_dict[pdid_list[i]]['Diabetes']) == 'nan':
            diab_list.append(0)
        else:
            diab_list.append(survival_dict[pdid_list[i]]['Diabetes'])
        height_list.append(body_aligned[pdid_list[i]]['height'][0])
    age_arr = sklearn.preprocessing.StandardScaler().fit_transform(np.array(age_list).reshape(-1, 1))
    height_arr = sklearn.preprocessing.StandardScaler().fit_transform(np.array(height_list).reshape(-1, 1))
    gender_arr = np.array(gender_list).reshape(-1, 1)
    diab_arr = np.array(diab_list).reshape(-1, 1)
    # static 原始数据
    age_list_raw = np.array(age_list).reshape(-1, 1)
    gender_list_raw = np.array(gender_list).reshape(-1, 1)
    diab_list_raw = np.array(diab_list).reshape(-1, 1)
    height_list_raw = np.array(height_list).reshape(-1, 1)
    static_dataset_raw = list(np.hstack([age_list_raw, gender_list_raw, diab_list_raw, height_list_raw]))
    # 662
    # 保存static 原始数据
    static_left = [static_dataset_raw[i] for i in indexs]
    pickle.dump(static_left,open(data_path + 'raw_static_4_662','wb'))

    # 656
    # 保存static 原始数据

    pdid_669 = pdid_list
    static_left_index = [pdid_669.index(id) for id in pdid_656]
    static_left = [static_dataset_raw[i] for i in static_left_index]
    pickle.dump(static_left,open(data_path + 'raw_static_4_656','wb'))

    # 669
    # 保存static 原始数据
    static_left = static_dataset_raw
    pickle.dump(static_left,open(data_path + 'raw_static_4_669','wb'))
    print(f'static_dataset_raw[0] : {static_dataset_raw[0]}')
    print(f'len(static_left) : {len(static_left)}')

    #验证scaler的正确性
    logger.info('check the scalers')
    print('如果一致，说明scaler正确')
    print(f'static_dataset[94] : {static_dataset[94]}')
    print(f'static_dataset_raw[94] : {static_dataset_raw[94]}')
    print(f"dump_scalers['age'].transform(static_dataset_raw[94][0].reshape(-1,1)) : {dump_scalers['age'].transform(static_dataset_raw[94][0].reshape(-1,1))}")

    # 验证x_656和x_662的不同
    # 虽然发现了不同但是没关系，我们可以直接通过id拿到原始数据
    logger.info('find difference between x_656 and x_662')
    pdid_662 = pdid_list2
    x_662 = x2
    print(len(x_656))
    print(len(x2))
    print(f"in 662 but not in 656 :{set(pdid_list2).difference(set(pdid_656))}")
    print(f"in 656 but not in 662 :{set(pdid_656).difference(set(pdid_list2))}")
def output_visit(pdid,static,raw_x,date,filepath,filename):
    visit_labels = gconst.get_labels_visit()
    static_labels = gconst.get_labels_static()
    result_dict = {'id':[],'date':[]}
    for key in visit_labels:
        result_dict[key] = []
    for key in static_labels:
        result_dict[key] = []
    for i,id in enumerate(pdid):
        data_len = len(raw_x[i])
        result_dict['id'] += [id]*data_len
        result_dict['date'] += date[i]
        for j,key in enumerate(visit_labels):
            result_dict[key]+=list(raw_x[i][:,j])
        for j,key in enumerate(static_labels):
            result_dict[key] += [static[i][j]]*data_len
        
    result_df = pd.DataFrame(result_dict)
    with pd.ExcelWriter(f'{filepath}{filename}') as writer:
        result_df.to_excel(writer,index=False,header=True)
    logger.debug(f'result saved at {filepath}{filename}')
    
def pick_day():
    '''get the birth day, admission day and so on [birthday,admission_day,age,death,death_day,death_age]'''

    args = parse_args()
    # load data
    origin_data_path = args.origin_data_path
    data_path = args.data_path # './data/healthier/'
    result_path = args.result_path
    pdid_656 = pickle.load(open(data_path+'656_pdid','rb'))
    date_656 = pickle.load(open(data_path+'656_date','rb'))
    static_656 = pickle.load(open(data_path+'raw_static_4_656','rb'))
    survival_dict = pickle.load(
        open(origin_data_path+'survival', 'rb'))
    days = []
    for index,id in enumerate(pdid_656):
        
        admission_day = date_656[pdid_656.index(id)][0]
        age = static_656[index][0]
        # birthday
        birthday = survival_dict[id]['Birth_date']

        if str(birthday)==str("nan") or id == 1206 or id==1207 or id ==1205:
            delta = datetime.timedelta(days=int(365*float(age)))
            birthday_obj = datetime.datetime.strptime(admission_day,'%Y/%m/%d') - delta
            birthday = birthday_obj.strftime('%Y/%m/%d')
            logger.debug(f"no brithday or birthday are wrong: {id} {birthday} {age}  {admission_day}")
        # age 
        if id == 1011 :
            age = (datetime.datetime.strptime(admission_day,'%Y/%m/%d') - datetime.datetime.strptime(birthday,'%Y/%m/%d')).days/365
            logger.debug(f"age are wrong: {id} {birthday} {age}  {admission_day}")

        #death day
        death = survival_dict[id]['Death']
        if(death == 1 and (id!=609 and id!=819)):
            death_day = survival_dict[id]['Death_date']
            death_age = survival_dict[id]['Death_age']
    
        else:
            death_day = date_656[index][-1]
            death_age = (datetime.datetime.strptime(death_day,'%Y/%m/%d')-datetime.datetime.strptime(birthday,'%Y/%m/%d')).days/365
        if(float(death_age)<float(age)):
            death_age = (datetime.datetime.strptime(death_day,'%Y/%m/%d')-datetime.datetime.strptime(birthday,'%Y/%m/%d')).days/365
        days.append([birthday,admission_day,age,death,death_day,death_age])
        if (id == 819 or id == 609):
            logger.debug(f"still have visit after death day: { id } {[birthday,admission_day,age,death,death_day,death_age]}")
    file_name = f'{data_path}656_days'
    pickle.dump(days,open(file_name,'wb'))
    id_days = [[pdid_656[index]] + i for index,i in enumerate(days)]
    df = pd.DataFrame(id_days,columns=['id']+gconst.get_labels_days())
    with pd.ExcelWriter(f'{result_path}days.xlsx') as writer:
        df.to_excel(writer,index=False)
    logger.debug(f'result saved at {file_name}')
def pic_date(pdid,date,result_path):
    '''get the total visit time for patients'''
    all_date_sub = [] # 患者总就诊时间
    for cur_date in date:
        date1 = datetime.datetime.strptime(cur_date[0],'%Y/%m/%d')
        date2 = datetime.datetime.strptime(cur_date[-1],'%Y/%m/%d')
        sub = (date2 - date1).days
        all_date_sub.append(sub)
    # print(all_date_sub)
    # all_date_sub.sort()
    # print(all_date_sub)
    result_dict = {'pdid':pdid,'visit_date_all':all_date_sub}
    with open(f'{result_path}visit_date_all.csv','w') as f:
        pd.DataFrame(result_dict).to_csv(f,index=False)
    logger.debug(f'result saved at {result_path}visit_date_all.csv')

    pic_name = f'{result_path}pic/visit_all_date_sub.png'
    plt.figure()

    sns.displot(all_date_sub,kde=True)
    # plt.title("患者总就诊时间分布")
    plt.savefig(pic_name)
    logger.debug(f'result saved at {pic_name}')
def pic_age(pdid,ages,result_path):
    '''get the distribution of patients' ages'''
    plt.figure()
    pic_name = f'{result_path}pic/visit_age.png'
    sns.displot(x=ages)
    plt.savefig(pic_name)
    logger.debug(f'result saved at {pic_name}')



def main():
    args = parse_args()
    data_path = args.data_path
    result_path = args.result_path
    origin_data_path = args.origin_data_path
    os.makedirs(data_path,exist_ok=True)
    # pdid_656 = pickle.load(open(f'{data_path}656_pdid','rb'))
    # date = pickle.load(open(f'{data_path}656_date','rb'))

    if args.task_pick_visit:
        logger.info('task_pick_visit: start.')
        pick_visit()
        logger.info('task_pick_visit: end.')
    if args.task_output_visit:
        logger.info('task_output_visit: start.')
        raw_x = pickle.load(open(f'{data_path}raw_x_17_656','rb'))
        raw_static = pickle.load(open(f'{data_path}raw_static_4_656','rb'))
        output_visit(pdid_656,raw_static,raw_x,date,result_path,'visit.xlsx')
        logger.info('task_output_visit: end.')
    if args.task_pick_day:
        logger.info('task_pick_day: start.')
        pick_day()
        logger.info('task_pick_day: end.')
    if args.task_pic_date:
        logger.info('task_pic_date: start.')
        pic_date(pdid_656,date,result_path)
        logger.info('task_pic_date: end.')
    if args.task_pic_age:
        static_raw = pickle.load(open(f'{data_path}raw_static_4_656','rb'))
        static = np.array(static_raw)
        logger.info('task_pic_age: start.')
        age = static[:,gconst.get_labels_static().index('age')]
        pic_age(pdid_656,age,result_path)
        logger.info('task_pic_age: end.')
def parse_args() -> argparse.Namespace:    

    '''Some mutable parameters'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_data_path', type=str, default='./data/origin/ckd/', help='Original data path')
    parser.add_argument('--data_path', type=str, default='./data/healthier/', help='Processed data path')
    parser.add_argument('--result_path', type=str, default='./result_sci/', help='Result path')

    parser.add_argument('--task_pick_visit', default=False, action='store_true', help='Extract visits from original data and save versions 656, 662, 669, raw, and scalers')
    parser.add_argument('--task_pick_day', default=False, action='store_true', help='Extract birthday, deathday, etc. from original data and save version 656')
    parser.add_argument('--task_output_visit', default=False, action='store_true', help='Output extracted visits, including dates and static data')
    parser.add_argument('--task_pic_date', default=False, action="store_true", help="Output image, total visit time for patients")
    parser.add_argument('--task_pic_age', default=False, action="store_true", help="Output image, distribution of patients' ages")

    return parser.parse_args()

if __name__ == '__main__':
    RANDOM_SEED = 12345
    np.random.seed(RANDOM_SEED) #numpy
    random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = '0'
    main()
