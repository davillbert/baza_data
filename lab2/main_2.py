import os

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
''' 
for k in data_map:
    cur_fig = go.Figure()
    cur_fig.add_trace(go.Scatter(x=time_map[k], y=data_map[k], name=f'Ref Torque: {k}'))
    # cur_fig.add_hline(data_map[k].std())
    cur_fig.update_layout(legend_orientation="v",
                      legend=dict(x=0.9, xanchor="right", y=0.9, yanchor='top',
                        font=dict(size=16)),
                      title=f'Ref Torque: {k}',
                      xaxis_title='Время, мс',
                      yaxis_title="Момент импульса, м^2 кг/с",
                      margin=dict(l=0, r=0, t=30, b=0))
    cur_fig.show()
'''
std_fig = go.Figure()
std_fig.add_trace(go.Scatter(x=list(data_map.keys()), y=std_arr, name='STD'))
std_fig.update_layout(title='Std',
                  xaxis_title='Ref Torque',
                  yaxis_title="Std",
                  margin=dict(l=0, r=0, t=30, b=0))
# std_fig.show()
std_fig.write_image('std(ref_torque).png')
print('Hello')