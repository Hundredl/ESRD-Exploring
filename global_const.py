'''
save some global const
'''
from typing import Dict, List
from unittest import result
import numpy as np

def get_ui():
    return {'ca':2000,'p':3500,'fe':42,'zn':40,'cu':8,'se':400,'vitaminA':3000,'nicotinic':35,'ascorbic':2000}
def get_labels_visit():
    labels = ['Cl', 'CO2CP', 'WBC', 'Hb', 'Urea', 'Ca', 'K', 'Na', 'Scr', 'P', 'Albumin', 'hs-CRP', 'Glucose']+ ['Appetite','Weight','SBP','DBP']
    return labels
def get_labels_static():
    labels2 = ['age','gender','diab','height']
    return labels2
def get_labels_diet():
    diet_labels = ['water', 'protein', 'fat', 'carbohydrate', 'Calories', 'df', 'k', 'na', 'mg', 'ca', 'p', 'fe', 'zn', 'cu', 'mn', 'se', 'retinol', 'vitaminA', 'carotene', 'vitaminE', 'thiamine', 'riboflavin', 'nicotinic', 'ascorbic']
    return diet_labels
def get_labels_diet_g():
    '''the labels of diet_g'''
    diet_labels = ['water', 'protein', 'fat', 'carbohydrate', 'df']
    return diet_labels
def get_labels_cox_multi_ok():
    return ['SBP','Scr','P','Albumin','age']
def get_labels_days():
    return ['birthday','admission_day','age','death','death_day','death_age']

def get_labels_statistics_visit():
    labels_visit = get_labels_visit()
    labels_static = get_labels_static()
    return labels_visit + labels_static + ['bmi','gfr']

def get_labels_statistics_diet():
    labels_diet = get_labels_diet()
    labels_statistics_diet = ['date'] + labels_diet + ['dpi','dei']
    return labels_statistics_diet

def get_labels_dict_statistics_visit_for_tab():
    return {'Albumin': 'Albumin', 'age': 'Age', 'gender': 'Gender', 'height': 'Height', 'Weight': 'Weight', 'bmi': 'BMI', 'diab': 'Diab', 'SBP': 'SBP', 'DBP': 'DBP', 'WBC': 
    'WBC', 'Hb': 'Hb', 'Urea': 'Urea', 'Scr': 'Scr', 'K': 'K', 
    'Na': 'Na', 'Cl': 'Cl', 'Ca': 'Ca', 'P': 'P', 'hsCRP': 'Hs-CRP', 
    'Glucose': 'Glucose', 'CO2CP': 'CO2CP', 'Appetite': 'Appetite', 'gfr': 'GFR'}
def get_labels_dict_statistics_diet_for_tab():
    return {'date': 'Date', 'water': 'Water', 'protein': 'Protein', 'fat': 'Fat', 'carbohydrate': 'Carbohydrate', 'Calories': 'Calories', 'df': 'Df', 'k': 'K', 'na': 'Na', 'mg': 'Mg', 'ca': 'Ca', 'p': 'P', 'fe': 'Fe', 'zn': 'Zn', 'cu': 'Cu', 'mn': 'Mn', 'se': 'Se', 'retinol': 'Retinol', 'vitaminA': 'VitaminA', 'carotene': 'Carotene', 'vitaminE': 'VitaminE', 'thiamine': 'Thiamine', 'riboflavin': 'Riboflavin', 'nicotinic': 'Nicotinic', 'ascorbic': 'Ascorbic', 'dpi': 'DPI', 'dei': 'DEI'}
def get_labels_statistics_diet_for_pic():
    # ['water', 'protein', 'fat', 'carbohydrate', 'Calories', 'df', 'k', 'na', 'mg', 'ca', 'p', 
    # 'fe', 'zn', 'cu', 'mn', 'se', 'retinol', 'vitaminA', 'carotene', 'vitaminE', 'thiamine', 'riboflavin',
    #  'nicotinic', 'ascorbic']
    return ['Date','Water','Protein (g/d)','Fat','Carbohydrate','Calories','Df','K','Na',
            'Mg','Ca','P','Fe','Zn','Cu','Mn','Se','Retinol','VitaminA','Carotene','VitaminE','Thiamine',
            'Riboflavin','Nicotinic','Ascorbic','DPI','DEI',]
def get_labels_statistics_diet_for_pic_with_unit():
    # ['water', 'protein', 'fat', 'carbohydrate', 'Calories', 'df', 'k', 'na', 'mg', 'ca', 'p', 
    # 'fe', 'zn', 'cu', 'mn', 'se', 'retinol', 'vitaminA', 'carotene', 'vitaminE', 'thiamine', 'riboflavin',
    #  'nicotinic', 'ascorbic']
    return ['Date','Water (g/d)','Protein (g/d)','Fat (g/d)','Carbohydrate (g/d)','Calories (kcal/d)','Df (g/d)',
                  'K (mg/d)','Na (mg/d)',
            'Mg (mg/d)','Ca (mg/d)','P (mg/d)','Fe (mg/d)','Zn (mg/d)','Cu (mg/d)',
                  'Mn (mg/d)','Se (mg/d)','Retinol(ug/d)','VitaminA (ugRAE/d)','Carotene (ug/d)','VitaminE (mg/d)','Thiamine (mg/d)',
            'Riboflavin (mg/d)','Nicotinic (mg/d)','Ascorbic (mg/d)','DPI (g/kg/d)','DEI (kcal/kg/d)']
#index 
def get_test_index():
    return [235, 609, 624, 545, 643, 221, 255, 351, 487, 360, 558, 
            648, 590, 246, 561, 270, 26, 581, 78, 334, 497, 501, 49, 
            23, 94, 414, 316, 252, 367, 446, 269, 115, 610, 445, 371, 2, 223, 559, 
            388, 431, 208, 382, 164, 483, 574, 297, 122, 552, 86, 171, 229, 463, 
            0, 282, 518, 615, 236, 386, 378, 34, 311, 199, 190, 620, 326]
def get_train_index(all_len=656):
    '''get the index of pdid for training'''
    all_index = list(range(all_len))
    test_index = get_test_index()
    result = [i for i in all_index if i not in set(test_index)]
    return result
def get_test_pdid(pdid):
    return [pdid[i] for i in get_test_index()]
def get_train_pdid(pdid):
    return [pdid[i] for i in range(len(pdid)) if i not in set(get_test_index())]


# rounds
def get_data_qs(data,type='q5'):
    position = (0,20,40,60,80,100)
    if type=='q5':
        position = (0,20,40,60,80,100)
    if type=='q3':
        position = (0,33,66,100)
    if type=='q2':
        position = (0, 50, 100)
    res = np.percentile(data,position)
    return res

def get_data_rounds_from_qs(qs):
    '''generate the rounds from qs'''
    age_rounds = []
    for i in range(len(qs) - 1):
        age_rounds.append((qs[i],qs[i + 1]))
    return age_rounds

def get_data_rounds_str(age_rounds):
    age_rounds_str = [f'[{i[0]:.1f},{i[1]:.1f})' for i in age_rounds]
    if(type(age_rounds[0][0])==type(1.2)):
        age_rounds_str = [f'[{i[0]:.1f},{i[1]:.1f})' for i in age_rounds]
    return age_rounds_str

def get_data_rounds_indexs(datas,data_rounds):
    '''change the continuous data to discrete data'''
    indexs = []
    for cur_data in datas:
        for index,data_round in enumerate(data_rounds):
            # print(index)
            if data_round[0]<=cur_data<data_round[1] + 0.0000001:
                indexs.append(index)
                break
    return indexs

def get_data_rounds(data,type = 'q3'):
    '''return the discrete range of data'''
    qs = get_data_qs(data,type)
    data_rounds = get_data_rounds_from_qs(qs)
    return data_rounds