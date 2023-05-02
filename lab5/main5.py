import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from funcs import (cut_df, cut_df_alot, df_date_and_time_to_time_in_ms,
                   df_from_csv_w_cols, get_ampl_of_vec, get_std_of_vec,
                   get_vec_abs_from_components, hhmmss2ms)

# КРИТЕРИЙ ВЫБОРА ДАННЫХ ДЛЯ УСЛОВИЯ |dir.y| < criteria. 
# ПРИ ВЫПОЛНЕНИИ УСЛОВИЯ БУДЕТ СЧИТАТЬСЯ, ЧТО ВЕКТОР 
# НАПРАВЛЕНИЯ НА СОЛНЦЕ ЛЕЖИТ В ПЛОСКОСТИ Oxz
criteria = 0.01


if os.getcwd().split('\\')[-1] == 'lab5':
    path = ['data/CF0_СД 0x0CF0/22A6_device.regularTelemetry.csv']
else:
    path = ['lab5/data/CF0_СД 0x0CF0/22A6_device.regularTelemetry.csv']

path = ['lab5/data/CF0_СД 0x0CF0/22A6_device.regularTelemetry.csv']

columns = [['Date and Time ISO', 'Date and Time', 'direction.x', 'direction.y', 'direction.z']]

xyz_cols = [['direction.x', 'direction.y', 'direction.z']]

t_cols = ['Date and Time']

vec_names = ['S']

START_TIMEs = ['12:14:00']

time_brackets = [[ # height
    ['12:14:00', '12:17:00'],
    ['12:17:00', '12:19:00'],
    ['12:19:00', '12:25:00']
    ]
]

def main(path, columns, xyz_cols, t_col, vec_name, START_TIME, time_brackets, criteria):
    for i in range(len(path)):
        df = df_from_csv_w_cols(path[i], columns[i])
        df = df_date_and_time_to_time_in_ms(df, t_col=t_col[i], START_TIME=START_TIME[i])
        # S_df = get_vec_abs_from_components(df, xyz_cols[i], t_col=t_col[i], vec_name=vec_name[i])
        df = cut_df(df, -1 * criteria, criteria, t_col='direction.y')
        dfs_array = cut_df_alot(df, time_brackets[i], START_TIME[i], t_col[i])
    
        for df_i in dfs_array:
            
            std_array.append(get_std_of_vec(df_i, vec_name[i]))
            ampl_array.append(get_ampl_of_vec(df_i, vec_name[i]))

    print('End')


main(path, columns, xyz_cols, t_cols, vec_names, START_TIMEs, time_brackets, criteria)

