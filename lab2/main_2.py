import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from prev_funcs import change_time, df_date2time, hhmmss2ms
from prettytable import PrettyTable

#os.chdir('lab2') os.getcwd()+'\HistoryLog/


START_TIME = '12:19:00.000'
df_torque_data = pd.read_csv('HistoryLog/1210_УДМ_Y/2101_rw.torque.csv', delimiter=';')

time_torque_df = pd.DataFrame(pd.concat([df_date2time(df_torque_data['Date and Time'], START_TIME), df_torque_data['torque']], axis=1),
                                columns=['Date and Time', 'torque'])
time_torque_df.columns = ['Время, мс', 'Момент импульса, м^2 кг/с']
# time_torque_df = change_time(time_torque_df)
# the_fig = px.line(time_torque_df, x='Время, мс', y='Момент импульса, м^2 кг/с', markers=False)
# the_fig.show()

START_PID0_TIME = '12:53:29.470'
FINISH_PID0_TIME = '12:54:25.749'

START_PID1_TIME = '12:54:25.749'
FINISH_PID1_TIME = '12:59:23.750'

START_PID2_TIME = '12:59:23.750'
FINISH_PID2_TIME = '13:01:50.466'

START_PID3_TIME = '13:01:50.466'
FINISH_PID3_TIME = '13:06:29.069'

START_PID4_TIME = '13:06:29.069'
FINISH_PID4_TIME = '13:11:19.824'



df_torque_pid_data = pd.read_csv('HistoryLog/1210_УДМ_Y/2101_rw.torque.csv', delimiter=';')
time_for_pid_torque_df = pd.DataFrame(pd.concat([df_date2time(df_torque_pid_data['Date and Time'], START_PID0_TIME),
                                                 df_torque_pid_data['torque']], axis=1), columns=['Date and Time', 'torque'])

time_for_pid_torque_df.columns = ['Время, мс', 'Момент импульса, м^2 кг/с']

# ref_torq: (start_time, end_time)
torq_map = {
        "1e-5": (566662,717295),
        "1e-4": (770130,890000),
        # "-1e-4": (,),
        "5e-4": (1093843,1135696),
        # "-5e-4": (,),
        "1e-3": (1362862,1383925),
        # "-1e-3": (,),
        "2e-3": (1608720,1618993),
        # "-2e-3": (,),
        "4e-3": (1825731,1830937)
        # "-4e-3": (,)
}
# ref_torq: np.array(torques)
data_map = {
        k: np.array(time_torque_df[
            (time_torque_df['Время, мс'] > v[0])
            & (time_torque_df['Время, мс'] < v[1])
        ]['Момент импульса, м^2 кг/с'])
        for k, v in torq_map.items()
}

# ref_torq: std
std_map = {k: data_map[k].std() for k in data_map}
std_arr = [data_map[k].std() for k in data_map]
# ref_torq: np.array(times)
time_map = {
        k: np.array(time_torque_df[
            (time_torque_df['Время, мс'] > v[0])
            & (time_torque_df['Время, мс'] < v[1])
        ]['Время, мс'])
        for k, v in torq_map.items()
}

# pid0 - P:0.1  I:0.01  D:0
# pid1 - P:0.01 I:0.1   D:0
# pid2 - P:0.01 I:0.001 D:0
# pid3 - P:0.01 I:0.01  D:0.01
# pid4 - P:0.01 I:0     D:0.01

pid_torq_map = {
        "pid0": (0,56279),
        "pid1": (56279,354280),
        "pid2": (354280,500996),
        "pid3": (500996,779599),
        "pid4": (779599,47479824)

}

data_pid_map = {
    k: np.array(time_for_pid_torque_df[
                    (time_for_pid_torque_df['Время, мс'] > v[0])
                    & (time_for_pid_torque_df['Время, мс'] < v[1])
                    ]['Момент импульса, м^2 кг/с'])
    for k, v in pid_torq_map.items()
}


std_pid_map = {k: data_pid_map[k].std() for k in data_pid_map}
std_pid_arr = [data_pid_map[k].std() for k in data_pid_map]
# ref_torq: np.array(times)
print(std_pid_arr)

time_pid_map = {
        k: np.array(time_torque_df[
            (time_torque_df['Время, мс'] > v[0])
            & (time_torque_df['Время, мс'] < v[1])
        ]['Время, мс'])
        for k, v in pid_torq_map.items()
}



fig, ax = plt.subplots()
plt.title("Среднеквадратичное отклонение")
plt.xlabel("Требуемое значение момента, м^2 кг/с")
plt.ylabel("Среднеквадратичное отклонение")
plt.grid(True, axis='both')
ax.plot(list(data_map.keys()), std_arr, marker='o', markersize=8, color='black', linestyle='solid')
# plt.show()
fig.savefig('std(refT).png', dpi=150)


fig.set_figheight(10)
fig.set_figwidth(10)
fig.savefig('std(refT_newsize).png', dpi=150)


fig1, ax1 = plt.subplots()
plt.title("Среднеквадратичное отклонение")
plt.xlabel("Требуемое значение момента, м^2 кг/с")
plt.ylabel("Среднеквадратичное отклонение в зависимости от PID")

fig1.set_figheight(10)
fig1.set_figwidth(10)
plt.grid(True, axis='both')

ax1.plot(list(data_pid_map.keys()), std_pid_arr, marker='o', markersize=8, color='black', linestyle='solid')
fig1.savefig('std(refT1).png', dpi=150)

th = ['number', 'P', 'I', 'D', 'CKO']
td = ['0', '0.1', '0.01',  '0', std_pid_arr[0],
      '1', '0.01', '0.1',  '0', std_pid_arr[1],
      '2', '0.01', '0.001',  '0', std_pid_arr[2],
      '3', '0.01', '0.01',  '0.01', std_pid_arr[3],
      '4', '0.01', '0',  '0.01', std_pid_arr[4]]

columns = len(th)

table = PrettyTable(th)

td_data = td[:]

while td_data:
    table.add_row(td_data[:columns])
    td_data = td_data[columns:]

print(table)


print('Hello, bruh')