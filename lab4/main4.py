# from lab2.prev_funcs import hhmmss2ms, df_date2time, change_time
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from funcs import (cut_df, cut_df_alot, df_date_and_time_to_time_in_ms,
                   df_from_csv_w_cols, get_ampl_of_vec, get_std_of_vec,
                   get_vec_abs_from_components, hhmmss2ms)

path = ['lab4/HistoryLog_2023-03-21_11-34-12/1008_Магнитометр/2903_device.regular1.csv']
columns = [['Date and Time', 'regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z']]
xyz_cols = [['regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z']]
t_cols = ['Date and Time']
vec_names = ['B']

START_TIMEs = ['11:39:00']
time_brackets = [[
    ['11:39:09', '11:39:29'],
    ['11:40:04', '11:40:24'],
    ['11:40:39', '11:40:59'],
    ['11:41:00', '11:41:20'],
    ['11:41:31', '11:41:51']
    ]
]
param_names = ['Angular Rate']
param_values = [['200', '400', '600', '800', '1000']]
plot_titles = ['Влияние скорости вращения УДМ на показания магнитометра.']

def main(path, columns, xyz_cols, t_col, vec_name, START_TIME, time_brackets, param_name, param_values, plot_titles):
    for i in range(len(path)):
        df = df_from_csv_w_cols(path[i], columns[i])
        df = df_date_and_time_to_time_in_ms(df, t_col=t_col[i], START_TIME=START_TIME[i])
        B_df = get_vec_abs_from_components(df, xyz_cols[i], t_col=t_col[i], vec_name=vec_name[i])
    
        B_dfs_array = cut_df_alot(B_df, time_brackets[i], START_TIME[i], t_col[i])
    
        std_array = []
        ampl_array = []
        for df_i in B_dfs_array:
            std_array.append(get_std_of_vec(df_i, vec_name[i]))
            ampl_array.append(get_ampl_of_vec(df_i, vec_name[i]))
    
    
        fig1 = plt.figure()
        fig1.suptitle(plot_titles[i])
        fig1.set_figheight(10)
        fig1.set_figwidth(10)
    
        ax1_1 = fig1.add_subplot(2,1,1)
        ax1_2 = fig1.add_subplot(2,1,2)
    
        ax1_1.plot(param_values[i], std_array, marker='o', markersize=8, color='black', linestyle='solid')
        ax1_1.set_title("Среднеквадратичное отклонение", fontsize=10)
        ax1_1.set(xlabel=param_name[i], 
                ylabel="Среднеквадратичное отклонение,\n нормированное на единицу")
        ax1_1.grid(True, axis='both')
    
        ax1_2.plot(param_values[i], ampl_array, marker='o', markersize=8, color='black', linestyle='solid')
        ax1_2.set_title("Амплитуда", fontsize=10)
        ax1_2.set(xlabel=param_name[i], 
                ylabel="Амплитуда $(max-min)/2$")
        ax1_2.grid(True, axis='both')
    
        fig1.savefig('lab4/СКО и амплитуда(omega).png', dpi=150)
        plt.show()

main(path, columns, xyz_cols, t_cols, vec_names, START_TIMEs, time_brackets, param_names, param_values, plot_titles)


print('Hello')
