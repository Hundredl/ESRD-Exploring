data_path_root='./esrd'
python ${data_path_root}/src/data_visit.py \
    --origin_data_path ${data_path_root}/origin/ckd/ \
    --data_path ${data_path_root}/processed/ \
    --result_path ${data_path_root}/result/ \
    --task_pick_visit \
    --task_pick_day


python ${data_path_root}/src/data_diet.py \
    --origin_data_path ${data_path_root}/origin/ckd/ \
    --data_path ${data_path_root}/processed/ \
    --result_path ${data_path_root}/result/ \
    --pick_nutrition_times 2 \
    --task_pick_diet 