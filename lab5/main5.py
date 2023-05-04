import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import math

from funcs import (cut_df, cut_df_alot, df_date_and_time_to_time_in_ms,
                   df_from_csv_w_cols, get_ampl_of_vec, get_std_of_vec,
                   get_vec_abs_from_components, hhmmss2ms, ms_to_time_str)

# КРИТЕРИЙ ВЫБОРА ДАННЫХ ДЛЯ УСЛОВИЯ |dir.y (или dir.x)| < criteria. 
# ПРИ ВЫПОЛНЕНИИ УСЛОВИЯ БУДЕТ СЧИТАТЬСЯ, ЧТО ВЕКТОР 
# НАПРАВЛЕНИЯ НА СОЛНЦЕ ЛЕЖИТ В ПЛОСКОСТИ Oxz (Oyz)
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

START_TIMEs = ['12:13:55', '12:34:55']

time_brackets = [[ # xz
    ['12:14:00', '12:15:00'], # не получилось найти интервал там просто прямая хз
    ['12:17:45', '12:18:20'],
    ['12:19:20', '12:20:30'],
    ['12:25:20', '12:25:40'],
    ['12:27:35', '12:28:00']
    ],
    [ # yz
    ['12:35:00', '12:35:35'],
    ['12:38:10', '12:38:50'],
    ['12:41:45', '12:42:30'],
    ['12:44:00', '12:44:50'],
    ['12:46:45', '12:47:30'],
    ['12:48:30', '12:49:15'],
    ['12:50:45', '12:51:15'],
    ['12:52:00', '12:53:00']
    ]
]

beta = [ # Углы, измеренные уровнем
    [20.26, 34.90, 27.30, 7.75, -3.7],
    [17.19, 10.13, -2.86, 39.30, -3.87, 6.62, 14.12, 24.32]]

alpha = [ # Углы падения Солнца, взятые с сайта sunearthtools.com
    [49.61, 49.64, 49.66, 49.68, 49.68],
    [49.65, 49.62, 49.58, 49.55, 49.49, 49.45, 49.41, 49.38]]


def main(path, columns, xyz_cols, t_col, vec_name, START_TIME, time_brackets, alpha, beta, criteria, sigmas):
    for i in range(len(path)):
        df = df_from_csv_w_cols(path[i], columns[i])
        df = df_date_and_time_to_time_in_ms(df, t_col=t_col[i], START_TIME=START_TIME[i])
        S_df = get_vec_abs_from_components(df, xyz_cols[i], t_col=t_col[i], vec_name=vec_name[i])
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

        ''' alpha = acos(z) в зависимости от T\
            ВРОДЕ БЫ ДАННЫЕ ВЫГЛЯДЯТ НОРМАЛЬНО, КРОМЕ ПЕРВОГО ЭКСПЕРИМЕНТА
            тогда можно юзать df_i из dfs_array как данные для отдельных экспериментов'''
        # alpha_measured = [math.degrees(math.acos(z)) for z in z_arr]
        # fig2 = plt.figure()
        # fig2.set_figheight(10)
        # fig2.set_figwidth(14)
        # fig2.add_gridspec(2, 2)
        # ax2 = fig2.add_subplot(1,1,1)
        # ax2.scatter(time_arr, alpha_measured, s=1, marker=',', color='blue')
        # ax2.set_xticks(time_arr, new_time_arr)
        # ax2.locator_params(axis='x', nbins=10)
        # plt.show()



        if i == 0:
            figure, axis = plt.subplots(2, 3)
            plot_name = '$|\\Delta\\alpha|(t), dir.y \\approx 0$'
            figure.suptitle(plot_name)
        elif i == 1:
            figure, axis = plt.subplots(2, 4)
            plot_name = '$|\\Delta\\alpha|(t), dir.x \\approx 0$'
            figure.suptitle(plot_name)

        figure.set_figheight(10)
        figure.set_figwidth(14)

        for j in range(len(dfs_array)):
            df_i = dfs_array[j]
            level_angle = beta[i][j]
            sun_angle = alpha[i][j] # alpha_ref
            z_dir_arr = list(np.array(df_i['direction.z']))
            alpha_arr = [(90 - math.degrees(math.acos(z_dir)) - level_angle) for z_dir in z_dir_arr]
            delta_alpha = [math.fabs(alph - sun_angle) for alph in alpha_arr]
            time_arr = list(np.array(df_i['Date and Time']))
            time_arr = [tau - (hhmmss2ms(time_brackets[i][j][0]) - hhmmss2ms(START_TIME[i])) for tau in time_arr]
            sum_sigma = math.sqrt(sigmas[0]) + math.sqrt(sigmas[1]) + math.sqrt(sigmas[2])
            sigma_arr = [sum_sigma for t in time_arr]

            if i == 0:
                m = j//3
                n = j%3
                axis[m,n].scatter(time_arr, delta_alpha, s=1)
                axis[m,n].plot(time_arr, sigma_arr, linewidth=1, color='black', linestyle='dashed')
                axis[m,n].set_ylim([0,1.5])
                axis[m,n].set_ylabel('$|\\Delta\\alpha|, grad$', fontsize=10, loc='top')
                axis[m,n].set_xlabel('t, мс', fontsize=10, loc='right')
                axis[m,n].set_title(f'experiment {j+1}\nbeta: {level_angle}\nsun angle: {sun_angle}', fontsize=10)

            elif i == 1:
                m = j//4
                n = j%4
                axis[m,n].scatter(time_arr, delta_alpha, s=1)
                axis[m,n].plot(time_arr, sigma_arr, linewidth=1, color='black', linestyle='dashed')
                axis[m,n].set_ylim([0,1.5])
                axis[m,n].set_ylabel('$|\\Delta\\alpha|, grad$', fontsize=10, loc='top')
                axis[m,n].set_xlabel('t, мс', fontsize=10, loc='right')
                axis[m,n].set_title(f'experiment {5+j+1}\nbeta: {level_angle}\nsun angle: {sun_angle}', fontsize=10)

                # print('1')
        plt.tight_layout()
        figure.savefig(f'lab5/Delta_alpha_t {i+1}.png', dpi=200)
        # plt.show()

            

    print('End')


main(path, columns, xyz_cols, t_cols, vec_names, START_TIMEs, time_brackets, alpha, beta, criteria, sigmas)

