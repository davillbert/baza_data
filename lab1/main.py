import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.optimize as spo
import numpy as np

# test comment

import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots


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

def line_interpol_func(tau, a,b):
  return a * tau + b

def create_interpol(measure, st_an_vel):
    arrt = []
    arry = []
    arrb = []
    k = 0
    for i in range(945):
        if int(measure['Start Angular Velocity'][i]) == st_an_vel:
            #print(measure['Angular Velocity'][i])
            arrt.append((measure['Время, мс'][i] - measure['Время, мс'][0])/1000)
            arry.append(measure['Angular Velocity'][i] / 500)
            if (measure['Время, мс'][i] - measure['Время, мс'][0]) != 0 and (measure['Angular Velocity'][i] / 500) > 0:
                arrb.append(-(np.log(measure['Angular Velocity'][i] / 500))/((measure['Время, мс'][i] - measure['Время, мс'][0])/1000))
            else:
                pass
            k = k + 1


    #popt, pcov = spo.curve_fit(exp_interpol_func, arrt, arry)
    popt1, _ = spo.curve_fit(exp_interpol_func, arrt,  arry)
    print("Exp arguments: ", popt1)

    popt2, _ = spo.curve_fit(line_interpol_func, arrt, arry)
    print("Line arguments: ", popt2)
    # using the optimal arguments to estimate new values

    ae,  be, ce = popt1[0], popt1[1], popt1[2]
    y_fit1 = []
    for t in arrt:
        y_fit1.append(exp_interpol_func(t, ae,  be, ce))
    #y_fit1 = ae * np.exp(-be * arrt)

    al, bl = popt2[0], popt2[1]
    y_fit2 = []
    for t in arrt:
        y_fit2.append(line_interpol_func(t, al, bl))
    #y_fit2 = al * np.exp(-bl * arrt)
    new_y = []
    for td in arrt:
        new_y.append(exp_interpol_func(td, ae, np.mean(arrb), ce))

    plt.plot(arrt, arry)
    plt.plot(arrt,  y_fit1, label="exp")
    plt.plot(arrt,  y_fit2, label="line")
    plt.plot(arrt, new_y, label="stupid exp")
    #print(popt[0])
    plt.xlabel('Angular Velocity')
    plt.ylabel('Время, мс')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)

    plt.savefig(f'approx_fit{st_an_vel}.png')
    ep = r'$e^{-{be}\\cdot\\tau}$'

    fig_app = go.Figure()
    fig_app.add_trace(go.Scatter(x=arrt, y=arry, name='Real  measure'))
    fig_app.add_trace(go.Scatter(x=arrt, y=y_fit2, name=f'$Line: y = { round(al,2)}\\cdot \\tau + { round(bl,2)} $'))
    fig_app.add_trace(go.Scatter(x=arrt, y=y_fit1, name=f'$Exp: y = { round(ae,2)}\\cdot e^{-round(be, 2)}\\tau + {round(ce,2)} $'))
    fig_app.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title=f'Approx Start Angular Velocity {st_an_vel}',
                      xaxis_title='$\\tau,  Время, мс$',
                      yaxis_title="y, Angular Velocity",
                      margin=dict(l=0, r=0, t=30, b=0))

    fig_app.show()



create_interpol(data, 500)
create_interpol(data, 1000)
create_interpol(data, 1500)
create_interpol(data, 2000)