import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import math

from funcs import (cut_df, cut_df_alot, df_date_and_time_to_time_in_ms,
                   df_from_csv_w_cols, get_ampl_of_vec, get_std_of_vec,
                   get_vec_abs_from_components, hhmmss2ms, ms_to_time_str)

# КРИТЕРИЙ ВЫБОРА ДАННЫХ ДЛЯ УСЛОВИЯ |dir.y| < criteria. 
# ПРИ ВЫПОЛНЕНИИ УСЛОВИЯ БУДЕТ СЧИТАТЬСЯ, ЧТО ВЕКТОР 
# НАПРАВЛЕНИЯ НА СОЛНЦЕ ЛЕЖИТ В ПЛОСКОСТИ Oxz
criteria = 0.05

# ПОГРЕШНОСТИ
sigma_SD = 0.2          # Датчика
sigma_level = 0.1       # Уровня
sigma_webModel = 0.01   # Модели Солнца на сайте
sigmas = [sigma_SD, sigma_level, sigma_webModel]
sum_sigma = math.sqrt(sigma_SD) + math.sqrt(sigma_level) + math.sqrt(sigma_webModel)

if os.getcwd().split('\\')[-1] == 'lab5':
    path = ['data/CF0_СД 0x0CF0/22A6_device.regularTelemetry.csv',
            'data/CF0_СД 0x0CF0/22A6_device.regularTelemetry.csv']
else:
    path = ['lab5/data/CF0_СД 0x0CF0/22A6_device.regularTelemetry.csv',
            'lab5/data/CF0_СД 0x0CF0/22A6_device.regularTelemetry.csv']

path = ['lab5/data/CF0_СД 0x0CF0/22A6_device.regularTelemetry.csv',
        'lab5/data/CF0_СД 0x0CF0/22A6_device.regularTelemetry.csv']

columns = [['Date and Time ISO', 'Date and Time', 'direction.x', 'direction.y', 'direction.z'],
            ['Date and Time ISO', 'Date and Time', 'direction.x', 'direction.y', 'direction.z']]

xyz_cols = [['direction.x', 'direction.y', 'direction.z'],
            ['direction.x', 'direction.y', 'direction.z']]

t_cols = ['Date and Time',
          'Date and Time']

vec_names = ['S', 'S']

START_TIMEs = ['12:12:00', '12:34:00']

time_brackets = [[ # xz
    ['12:14:00', '12:15:00'],
    ['12:17:00', '12:18:00'],
    ['12:19:00', '12:20:00'],
    ['12:25:00', '12:26:00'],
    ['12:27:00', '12:28:00']
    ],
    [ # yz
    ['12:35:00', '12:36:00'],
    ['12:38:00', '12:39:00'],
    ['12:41:00', '12:42:00'],
    ['12:44:00', '12:45:00'],
    ['12:46:00', '12:47:00'],
    ['12:48:00', '12:49:00'],
    ['12:50:00', '12:51:00'],
    ['12:52:00', '12:53:00']
    ]
]

def main(path, columns, xyz_cols, t_col, vec_name, START_TIME, time_brackets, criteria, sigmas):
    for i in range(len(path)):
        df = df_from_csv_w_cols(path[i], columns[i])
        df = df_date_and_time_to_time_in_ms(df, t_col=t_col[i], START_TIME=START_TIME[i])
        # S_df = get_vec_abs_from_components(df, xyz_cols[i], t_col=t_col[i], vec_name=vec_name[i])
        if i == 0:
            df = cut_df(df, -1 * criteria, criteria, t_col='direction.y') # Cut df using criteria
        elif i == 1:
            df = cut_df(df, -1 * criteria, criteria, t_col='direction.x') # Cut df using criteria
        dfs_array = cut_df_alot(df, time_brackets[i], START_TIME[i], t_col[i])

        time_arr = []
        x_arr = []
        y_arr = []
        z_arr = []
        for df_i in dfs_array:
            tmp = np.array(df_i[t_col[i]])
            time_arr.extend(iter(tmp))

            tmp = np.array(df_i[xyz_cols[i][0]])
            x_arr.extend(iter(tmp))

            tmp = np.array(df_i[xyz_cols[i][1]])
            y_arr.extend(iter(tmp))

            tmp = np.array(df_i[xyz_cols[i][2]])
            z_arr.extend(iter(tmp))
        # xyz_arr = [x_arr, y_arr, z_arr]
        
        new_time_arr = []
        for j in time_arr:
            new_time_arr.append(ms_to_time_str(j + hhmmss2ms(START_TIME[i]))[:-2])

        ''' X, Y, Z в зависимости от T'''
        # fig1 = plt.figure()
        # fig1.set_figheight(10)
        # fig1.set_figwidth(14)
        # fig1.add_gridspec(2, 2)
        # ax1_1 = fig1.add_subplot(1,1,1)
        # ax1_1.scatter(time_arr, x_arr, s=1, marker=',', color='blue')
        # ax1_1.scatter(time_arr, y_arr, s=1, marker=',', color='green')
        # ax1_1.scatter(time_arr, z_arr, s=1, marker=',', color='red')
        # ax1_1.set_xticks(time_arr, new_time_arr)
        # ax1_1.locator_params(axis='x', nbins=20)
        # plt.show()

        ''' alpha = acos(z или y) в зависимости от T\
            ВРОДЕ БЫ ДАННЫЕ ВЫГЛЯДЯТ НОРМАЛЬНО, КРОМЕ ПЕРВОГО ЭКСПЕРИМЕНТА
            тогда можно юзать df_i из dfs_array как данные для отдельных экспериментов'''
        alpha_measured = [math.degrees(math.acos(z)) for z in z_arr]
        fig2 = plt.figure()
        fig2.set_figheight(10)
        fig2.set_figwidth(14)
        fig2.add_gridspec(2, 2)
        ax2 = fig2.add_subplot(1,1,1)
        ax2.scatter(time_arr, alpha_measured, s=1, marker=',', color='blue')
        ax2.set_xticks(time_arr, new_time_arr)
        ax2.locator_params(axis='x', nbins=10)
        plt.show()

        # for df_i in dfs_array:
        #     # Обработать каждый эксперимент
        #     pass

    print('End')


main(path, columns, xyz_cols, t_cols, vec_names, START_TIMEs, time_brackets, criteria, sigmas)

