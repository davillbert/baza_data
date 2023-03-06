import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from prev_funcs import change_time, df_date2time, hhmmss2ms, df_date2time_interval

os.chdir('lab2')


START_TIME = '12:19:00.000'
df_torque_data = pd.read_csv(os.getcwd()+'\HistoryLog/1210_УДМ_Y/2101_rw.torque.csv', delimiter=';')

time_torque_df = pd.DataFrame(pd.concat([df_date2time(df_torque_data['Date and Time'], START_TIME), df_torque_data['torque']], axis=1),
                                columns=['Date and Time', 'torque'])
time_torque_df.columns = ['Время, мс', 'Момент импульса, м^2 кг/с']
# time_torque_df = change_time(time_torque_df)
# the_fig = px.line(time_torque_df, x='Время, мс', y='Момент импульса, м^2 кг/с', markers=False)
# the_fig.show()

START_PID0_TIME = '12:53:00.000'
FINISH_PID0_TIME = '12:54:25.749'

START_PID1_TIME = '12:54:25.749'
FINISH_PID1_TIME = '12:59:23.750'

START_PID2_TIME = '12:59:23.750'
FINISH_PID2_TIME = '13:01:50.466'

START_PID3_TIME = '13:01:50.466'
FINISH_PID3_TIME = '13:06:29.069'

START_PID4_TIME = '13:06:29.069'
FINISH_PID4_TIME = '13:11:19.824'

df_torque_data_for_pid = pd.read_csv('HistoryLog/1210_УДМ_Y/2101_rw.torque.csv', delimiter=';')

time_torque_data_for_pid_df = pd.DataFrame(pd.concat([df_date2time(df_torque_data_for_pid['Date and Time'], START_PID0_TIME), df_torque_data_for_pid['torque']], axis=1),
                               columns=['Date and Time', 'torque'])
time_torque_data_for_pid_df.columns = ['Время, мс', 'Момент импульса, м^2 кг/с']

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

# pid0 - 0.1 0.01 0
# pid1 - 0.01 0.1 0
# pid2 - 0.01 0.001 0
# pid3 - 0.01 0.01 0.01
# pid4 - 0.01 0 0.01

pid_torq_map = {
        "pid0": (46380000,46465749),
        "pid1": (46465749,46763750),
        "pid2": (46763750,46910466),
        "pid3": (46910466,47189069),
        "pid4": (47189069,47479824)

}

data_pid_map = {
        k: np.array(time_torque_data_for_pid_df[
            (time_torque_data_for_pid_df['Время, мс'] > v[0])
            & (time_torque_data_for_pid_df['Время, мс'] < v[1])
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



print('Hello, bruh')