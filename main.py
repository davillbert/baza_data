import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.optimize as spo
import numpy as np


# hh:mm:ss.msmsms  ->  m_sec
def hhmmss2ms(hms):
    return (int(hms[:2]) * 60 * 60 + int(hms[3:5])*60 + int(hms[6:8]))*1000 + int(hms[9:])

# YY_MM_DD hh:mm:ss.ms  ->  hh:mm:ss.msmsms  ->  m_sec
def df_date2time(df_data):
    for i in range(len(df_data)):
        df_data.at[i] = hhmmss2ms(df_data[i][9:]) - hhmmss2ms(START_TIME)
    return df_data

# time  ->  time - start_time
def change_time(df, st_end_tuple):
    tmp = df['Время, мс'].to_list()
    for i in range(len(tmp)):
        tmp[i] = tmp[i] - st_end_tuple[0]
    tmp = pd.DataFrame(pd.Series(tmp), columns=['Время, мс'])
    tmp_angR = df['Угловая скорость, об/мин']
    tmp_angR = pd.DataFrame(df['Угловая скорость, об/мин'], columns=['Угловая скорость, об/мин']).set_axis(range(len(tmp)))
    new_df = pd.concat([tmp, tmp_angR], axis=1, ignore_index=True)
    new_df.columns = ['Время, мс', 'Угловая скорость, об/мин']
    return new_df

# Returns df for plot
def create_data(change_time, to_measure_brake):
    start1, end1 = 662871, 677868
    start2, end2 = 730500, 757200
    start3, end3 = 810200, 846500
    start4, end4 = 903870, 948800

    st_ends = [(start1, end1), (start2, end2), (start3, end3), (start4, end4)]

    brake_dfs = [
        # 500 -> 0
        to_measure_brake[
            (to_measure_brake['Время, мс'] > start1)
            & (to_measure_brake['Время, мс'] < end1)
        ],
        # 1000 -> 0
        to_measure_brake[
            (to_measure_brake['Время, мс'] > start2)
            & (to_measure_brake['Время, мс'] < end2)
        ],
        # 1500 -> 0
        to_measure_brake[
            (to_measure_brake['Время, мс'] > start3)
            & (to_measure_brake['Время, мс'] < end3)
        ],
        # 2000 -> 0
        to_measure_brake[
            (to_measure_brake['Время, мс'] > start4)
            & (to_measure_brake['Время, мс'] < end4)
        ],
    ]

    times = []
    for i in range(len(brake_dfs)):
        brake_dfs[i] = change_time(brake_dfs[i], st_ends[i])
        brake_dfs[i].columns = ['Время, мс', 'Angular Velocity']
        times.append(len(brake_dfs[i]))

    colors = mk_colors_for_plot(times)
    time_vel = pd.concat([brake_dfs[0], brake_dfs[1], brake_dfs[2], brake_dfs[3]], axis=0, ignore_index=True)
    data = pd.concat([colors, time_vel], axis=1, ignore_index=True)

    data.columns = ['Start Angular Velocity', 'Время, мс', 'Angular Velocity']
    return data

def mk_colors_for_plot(times):
    to_be_color = []
    for j in range(len(times)):
        for _ in range(times[j]):
            if j == 0: to_be_color.append('500')
            elif j == 1: to_be_color.append('1000')
            elif j == 2: to_be_color.append('1500')
            elif j == 3: to_be_color.append('2000')
            else: print("No")

    return pd.Series(to_be_color, name='Start Angular Velocity')





START_TIME = '11:25:00.000'
# set_ang_rate_df = pd.read_csv('1210_УДМ_Y/21B0_rw.setAngularRate.csv', delimiter=';')
df_angRate_data = pd.read_csv('1210_УДМ/2100_rw.angularRate.csv', delimiter=';')


# time_setAngRate_df = pd.DataFrame(pd.concat([df_date2time(set_ang_rate_df['Date and Time']),set_ang_rate_df['rate']], axis=1), columns=['Date and Time', 'rate'])
time_AngRate_df = pd.DataFrame(pd.concat([df_date2time(df_angRate_data['Date and Time']), df_angRate_data['angRate']], axis=1),
                                columns=['Date and Time', 'angRate'])
time_AngRate_df.columns = ['Время, мс', 'Угловая скорость, об/мин']

to_measure_brake = time_AngRate_df[(time_AngRate_df['Время, мс'] > 636522) & (time_AngRate_df['Время, мс'] < 949000)]


data = create_data(change_time, to_measure_brake)

fig = px.line(
    data, x='Время, мс', y='Angular Velocity', color='Start Angular Velocity', markers=False
)

fig.write_image("fig1.png")
fig.show()

def exp_interpol_func(tau, a, b, c):
  return a * np.exp(-b * tau) + c

