# from lab2.prev_funcs import hhmmss2ms, df_date2time, change_time
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from funcs import (cut_df, cut_df_alot, df_date_and_time_to_time_in_ms,
                   df_from_csv_w_cols, get_ampl_of_vec, get_std_of_vec,
                   get_vec_abs_from_components, hhmmss2ms)

path = ['lab4/HistoryLog_2023-03-21_11-34-12/1008_Магнитометр/2903_device.regular1.csv',
        'lab4/HistoryLog_2023-03-21_11-52-52/1008_Магнитометр/2903_device.regular1.csv',
        'lab4/HistoryLog_2023-03-21_12-06-34/1008_Магнитометр/2903_device.regular1.csv',
        'lab4/HistoryLog_2023-03-21_12-25-22/1008_Магнитометр/2903_device.regular1.csv']

columns = [['Date and Time', 'regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z'],
           ['Date and Time', 'regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z'],
           ['Date and Time', 'regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z'],
           ['Date and Time', 'regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z']]

xyz_cols = [['regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z'],
            ['regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z'],
            ['regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z'],
            ['regular1_data.mag.x', 'regular1_data.mag.y', 'regular1_data.mag.z']]

t_cols = ['Date and Time',
          'Date and Time',
          'Date and Time',
          'Date and Time']

vec_names = ['B',
             'B',
             'B',
             'B']

START_TIMEs = ['11:39:00',
               '11:52:00',
               '12:07:00',
               '12:25:00']
time_brackets = [[ # omega
    ['11:39:09', '11:39:29'],
    ['11:40:04', '11:40:24'],
    ['11:40:39', '11:40:59'],
    ['11:41:00', '11:41:20'],
    ['11:41:31', '11:41:51']
    ],

    [ # L
    ['11:53:17', '11:53:27'],
    ['11:55:49', '11:55:59'],
    ['11:56:16', '11:56:26'],
    ['11:56:59', '11:57:09'],
    ['11:57:38', '11:57:48'],
    ['11:58:14', '11:58:24'],
    ['11:58:39', '11:58:49']
    ],
    
    [ # alpha
    ['12:07:55', '12:08:05'],
    ['12:08:57', '12:09:07'],
    ['12:09:21', '12:09:31'],
    ['12:10:01', '12:10:11'],
    ['12:10:36', '12:10:46'],
    ['12:11:16', '12:11:26'],
    ['12:11:48', '12:11:58'],
    ['12:12:44', '12:12:54']
    ],

    [ # height
    ['12:31:15', '12:31:25'],
    ['12:30:46', '12:30:56'],
    ['12:30:09', '12:30:19'],
    ['12:29:37', '12:29:38'],
    ['12:28:54', '12:29:04'],
    ['12:28:01', '12:28:11']
    ]
]
param_names = ['Angular Rate',
               'L, cm',
               'alpha, grad',
               'Height, cm']

param_values = [['200', '400', '600', '800', '1000'],
                ['9', '11', '13', '15', '19', '23', '27'],
                ['0', '45', '90', '135', '180', '225', '270', '315'],
                ['8', '10', '12', '15', '18', '21']]

plot_titles = ['Влияние скорости вращения УДМ на показания ММ.',
               'Влияние расстояния между УДМ и ММ на показания ММ',
               'Влияние угла поворота ММ относительно УДМ на показания ММ',
               'Влияние высоты ММ относительно УДМ на показания ММ']

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
        fig1.suptitle(plot_titles[i], fontsize=20)
        fig1.set_figheight(10)
        fig1.set_figwidth(10)
    
        ax1_1 = fig1.add_subplot(2,1,1)
        ax1_2 = fig1.add_subplot(2,1,2)
    
        ax1_1.plot(param_values[i], std_array, marker='o', markersize=8, color='black', linestyle='solid')
        ax1_1.set_title("Среднеквадратичное отклонение", fontsize=12)
        ax1_1.set_xlabel(param_name[i], loc='right')
        ax1_1.set(ylabel="Среднеквадратичное отклонение,\n нормированное на единицу")
        ax1_1.grid(True, axis='both')
    
        ax1_2.plot(param_values[i], ampl_array, marker='o', markersize=8, color='black', linestyle='solid')
        ax1_2.set_title("Амплитуда", fontsize=12)
        ax1_2.set(ylabel="Амплитуда $(max-min)/2$")
        ax1_2.set_xlabel(param_name[i], loc='right')
        ax1_2.grid(True, axis='both')
    
        fig1.savefig(f'lab4/{plot_titles[i]}.png', dpi=200)
        # plt.show()

main(path, columns, xyz_cols, t_cols, vec_names, START_TIMEs, time_brackets, param_names, param_values, plot_titles)


