import os

import pandas as pd
import plotly.express as px
from prev_funcs import change_time, df_date2time, hhmmss2ms

os.chdir('lab2')

START_TIME = '12:19:00.000'
df_torque_data = pd.read_csv(os.getcwd()+'\HistoryLog/1210_УДМ_Y/2101_rw.torque.csv', delimiter=';')

time_torque_df = pd.DataFrame(pd.concat([df_date2time(df_torque_data['Date and Time'], START_TIME), df_torque_data['torque']], axis=1),
                                columns=['Date and Time', 'torque'])
time_torque_df.columns = ['Время, мс', 'Момент импульса, м^2 кг/с']

the_fig = px.line(time_torque_df, x='Время, мс', y='Момент импульса, м^2 кг/с', markers=False)
the_fig.show()

# torq_map = {"1e-5": (489104,)
#             "1e-4": (770130,)}

print('Hello')