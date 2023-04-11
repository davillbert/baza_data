import pandas as pd


def hhmmss2ms(hms):
    '''
    hh:mm:ss.msmsms  ->  m_sec
    '''
    return (int(hms[:2]) * 60 * 60 + int(hms[3:5])*60 + int(hms[6:8]))*1000 + int(hms[9:])


def df_date2time(df_data, START_TIME):
    '''
    YY_MM_DD hh:mm:ss.ms  ->  hh:mm:ss.msmsms  ->  m_sec
    '''
    for i in range(len(df_data)):
        df_data.at[i] = hhmmss2ms(df_data[i][9:]) - hhmmss2ms(START_TIME)
    return df_data


def change_time(df, st_end_tuple):
    '''
    time  ->  time - start_time
    '''
    tmp = df['Время, мс'].to_list()
    for i in range(len(tmp)):
        tmp[i] = tmp[i] - st_end_tuple[0]
    tmp = pd.DataFrame(pd.Series(tmp), columns=['Время, мс'])
    tmp_angR = df['Момент импульса, м^2 кг/с']
    tmp_angR = pd.DataFrame(df['Момент импульса, м^2 кг/с'], columns=['Момент импульса, м^2 кг/с']).set_axis(range(len(tmp)))
    new_df = pd.concat([tmp, tmp_angR], axis=1, ignore_index=True)
    new_df.columns = ['Время, мс', 'Момент импульса, м^2 кг/с']
    return new_df
