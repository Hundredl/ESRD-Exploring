import pickle
data_path_origin = '/home/wyy/workspace/exp/data/esrd/origin/ckd/'
data_path_desensitized = '/home/wyy/workspace/exp/data/esrd/origin/desensitized/'
blood = pickle.load(open(f'{data_path_origin}blood','rb'))
body_dict = pickle.load(open(f'{data_path_origin}body_dict','rb'))
survival = pickle.load(open(f'{data_path_origin}survival','rb'))


for key in blood:
    blood[key]['name'] = f'patient{str(int(key))}'
for key in body_dict:
    body_dict[key]['name'] = f'patient{str(int(key))}'
for key in survival:
    survival[key]['Name'] = f'patient{str(int(key))}'

pickle.dump(blood,open(f'{data_path_desensitized}blood','wb'))
pickle.dump(body_dict,open(f'{data_path_desensitized}body_dict','wb'))
pickle.dump(survival,open(f'{data_path_desensitized}survival','wb'))
print(f'result saved at {data_path_desensitized}')