# from lab2.prev_funcs import hhmmss2ms, df_date2time, change_time
import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from funcs import (cut_df, cut_df_alot, df_date_and_time_to_time_in_ms,
                   df_from_csv_w_cols, get_ampl_of_vec, get_std_of_vec,
                   get_vec_abs_from_components, hhmmss2ms)

path = ['SD_HistoryLog_2023-03-14_14-32-45/C08_СД_1/22A6_device.regularTelemetry.csv',
        'SD_HistoryLog_2023-03-14_14-32-45/C10_СД_2/22A6_device.regularTelemetry.csv']


columns = [['Date and Time', 'direction.x', 'direction.y', 'direction.z', 'invalidFlag'],
           ['Date and Time', 'direction.x', 'direction.y', 'direction.z', 'invalidFlag']]


xyz_cols = [['direction.x', 'direction.y', 'direction.z'],
            ['direction.x', 'direction.y', 'direction.z']]

t_cols = ['Date and Time', 'Date and Time']

vec_names = ['Sdot', 'Sdot']

valid_cols = ['invalidFlag', 'invalidFlag']

START_TIMEs = ['14:32:53', '14:32:53']

time_brackets = [
    ['14:33:09', '14:36:00'],
    ['14:33:09', '14:36:00']
    ]

buf = ''
plot_titles = ['График 1', 'График 2']

def main(buf, path, columns, xyz_cols, t_col, START_TIME, time_brackets, plot_titles):
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    for i in range(len(path)):
        df = df_from_csv_w_cols(path[i], columns[i])
        df = df_date_and_time_to_time_in_ms(df, t_col=t_col[i], START_TIME=START_TIME[i])

        if i == 0:
            df1 = df
        if i == 1:
            df2 = df
    dfs = pd.DataFrame({'Date and Time S1': df1['Date and Time'], 'Date and Time S2': df2['Date and Time'], 'x S1': df1['direction.x'], 'y S1': df1['direction.y'], 'z S1': df1['direction.z'],  'x S2': df2['direction.x'], 'y S2': df2['direction.y'], 'z S2': df2['direction.z'],'F S1':df1['invalidFlag'],'F S2':df2['invalidFlag']})
    dfs_srt = dfs[abs(dfs['Date and Time S1'] - dfs['Date and Time S2']) < 22]

    dfs_srt = dfs_srt[dfs_srt['F S1'] == True]
    dfs_srt = dfs_srt[dfs_srt['F S2'] == True]

    dfs_new = pd.DataFrame(
        {'Date and Time': dfs_srt['Date and Time S1'],  'x S1': dfs_srt['x S1'],
         'y S1': dfs_srt['y S1'], 'z S1': dfs_srt['z S1'], 'x S2': dfs_srt['x S2'], 'y S2': dfs_srt['y S2'],
         'z S2': dfs_srt['z S2']})

    dfs_srt_ctn = dfs_new.drop_duplicates(subset=['x S1', 'x S2', 'y S1', 'y S2', 'z S1', 'z S2'])
    dfs_srt_ctn = dfs_srt_ctn.reset_index(drop=True)
    # print(dfs_srt_ctn)

    W_arr = []
    V_arr = []
    VWT = []
    for i in range(len(dfs_srt_ctn)):
        if i == (len(dfs_srt_ctn) - 1):
            break
        else:
            #print(dfs_srt_ctn.iloc[i])
            S1 = dfs_srt_ctn.iloc[i+1,1]*dfs_srt_ctn.iloc[i,1] + dfs_srt_ctn.iloc[i+1,2]*dfs_srt_ctn.iloc[i,2] + dfs_srt_ctn.iloc[i+1,3]*dfs_srt_ctn.iloc[i,3]
            S2 = dfs_srt_ctn.iloc[i + 1, 4] * dfs_srt_ctn.iloc[i, 4] + dfs_srt_ctn.iloc[i + 1, 5] * dfs_srt_ctn.iloc[i, 5] + dfs_srt_ctn.iloc[i + 1, 6] * dfs_srt_ctn.iloc[i, 6]
            if (abs(S1-S2) / max(S1,S2) < 0.1):
                s1km = np.dot([dfs_srt_ctn.iloc[i,1],dfs_srt_ctn.iloc[i,2],dfs_srt_ctn.iloc[i,3]],
                              [dfs_srt_ctn.iloc[i+1,1],dfs_srt_ctn.iloc[i+1,2],dfs_srt_ctn.iloc[i+1,3]])

                V =  np.array([[dfs_srt_ctn.iloc[i,1], dfs_srt_ctn.iloc[i,2], dfs_srt_ctn.iloc[i,3]],
                               np.cross([dfs_srt_ctn.iloc[i,1],
                                          dfs_srt_ctn.iloc[i,2],
                                          dfs_srt_ctn.iloc[i,3]],
                                         [dfs_srt_ctn.iloc[i+1,1],
                                           dfs_srt_ctn.iloc[i+1,2],
                                           dfs_srt_ctn.iloc[i+1,3]]),
                               np.subtract([round(s1km,3)*dfs_srt_ctn.iloc[i,1],
                                            round(s1km,3)*dfs_srt_ctn.iloc[i,2],
                                            round(s1km,3)*dfs_srt_ctn.iloc[i,3]],
                                           [dfs_srt_ctn.iloc[i+1,1],
                                            dfs_srt_ctn.iloc[i+1,2],
                                            dfs_srt_ctn.iloc[i+1,3]])])

                s2km = np.dot([dfs_srt_ctn.iloc[i,4],dfs_srt_ctn.iloc[i,5],dfs_srt_ctn.iloc[i,6]],
                            [dfs_srt_ctn.iloc[i+1,4],dfs_srt_ctn.iloc[i+1,5],dfs_srt_ctn.iloc[i+1,6]])
                W = np.array([[dfs_srt_ctn.iloc[i,4], dfs_srt_ctn.iloc[i,5], dfs_srt_ctn.iloc[i,6]],
                               np.cross([dfs_srt_ctn.iloc[i,4],
                                          dfs_srt_ctn.iloc[i,5],
                                          dfs_srt_ctn.iloc[i,6]],
                                         [dfs_srt_ctn.iloc[i+1,4],
                                           dfs_srt_ctn.iloc[i+1,5],
                                           dfs_srt_ctn.iloc[i+1,6]]),
                               np.subtract([dfs_srt_ctn.iloc[i,4],
                                            dfs_srt_ctn.iloc[i,5],
                                            dfs_srt_ctn.iloc[i,6]],
                                           [round(s2km,3)*dfs_srt_ctn.iloc[i+1,4],
                                            round(s2km,3)*dfs_srt_ctn.iloc[i+1,5],
                                            round(s2km,3)*dfs_srt_ctn.iloc[i+1,6]])])


                #print(V)
                V_arr.append(V)
                #print(W)
                W_arr.append(W)
                VWT.append(np.dot(V, W.T))
        #t = dfs_srt_ctn['x S1'].at[i+1]*dfs_srt_ctn['x S1'].at[i]
        # matrix = np.array([[1, 2, 3], [2, 5, 6], [6, 7, 4]])
        #round((dfs_srt_ctn.iloc[i,1]*dfs_srt_ctn.iloc[i+1,1]+dfs_srt_ctn.iloc[i,2]*dfs_srt_ctn.iloc[i+1,2]+dfs_srt_ctn.iloc[i,3]*dfs_srt_ctn.iloc[i+1,3]))*
#dfs_srt_ctn.iloc[i,4]*dfs_srt_ctn.iloc[i+1,4]+dfs_srt_ctn.iloc[i,5]*dfs_srt_ctn.iloc[i+1,5]+dfs_srt_ctn.iloc[i,6]*dfs_srt_ctn.iloc[i+1,6])*
        # print ("Матрица:\n", matrix)



        #tmp_dfs3.append(t) 1 4     2 5     3 6

    #(tmp_dfs3)
    #Scal = pd.DataFrame({'S1': S1, 'S2': S2})

    #print(Scal)
    #S1 = []
    #   S2 = []
    #for i in range(len(Scal)):
       # if i == (len(Scal) - 1):
            #break
        #else:
            #print(Scal.iloc[i,1])
            #if(abs(Scal.iloc[i,0] - Scal.iloc[i,1])/max(Scal.iloc[i,0], Scal.iloc[i,1]) < 0.1):
            #    S1.append(Scal.iloc[i,0])
          #      S2.append(Scal.iloc[i,1])
    #scal_filt = pd.DataFrame(
      #  {'S1': S1, 'S2': S2})

    #print(scal_filt)

    #print(V_arr)
    #print(W_arr)
    for i in VWT:
        buf += "======================\n"
        # print("======================")
        buf += f'{i}\n'
        # print(i)
    return buf


buf = main(buf, path, columns, xyz_cols, t_cols, START_TIMEs, time_brackets, plot_titles)


with open('out.txt', mode='w', encoding='utf-8') as file:
    file.write(buf)
    # file.write('ASAFSAFFSA')
