import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from prev_funcs import change_time, df_date2time, hhmmss2ms

os.chdir('lab2')

START_TIME = '12:19:00.000'
df_torque_data = pd.read_csv(os.getcwd()+'\HistoryLog/1210_УДМ_Y/2101_rw.torque.csv', delimiter=';')

time_torque_df = pd.DataFrame(pd.concat([df_date2time(df_torque_data['Date and Time'], START_TIME), df_torque_data['torque']], axis=1),
                                columns=['Date and Time', 'torque'])
time_torque_df.columns = ['Время, мс', 'Момент импульса, м^2 кг/с']
# time_torque_df = change_time(time_torque_df)
# the_fig = px.line(time_torque_df, x='Время, мс', y='Момент импульса, м^2 кг/с', markers=False)
# the_fig.show()

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

fig, ax = plt.subplots()
plt.title("Среднеквадратичное отклонение")
plt.xlabel("Требуемое значение момента, м^2 кг/с")
plt.ylabel("Среднеквадратичное отклонение")
plt.grid(True, axis='both')
ax.plot(list(data_map.keys()), std_arr, marker='o', markersize=8, color='black', linestyle='solid')
# plt.show()
fig.savefig('std(refT).png', dpi=150)

print('Hello')