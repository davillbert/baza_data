import pandas as pd
# hh:mm:ss.msmsms  ->  m_sec
def hhmmss2ms(hms):
    return (int(hms[:2]) * 60 * 60 + int(hms[3:5])*60 + int(hms[6:8]))*1000 + int(hms[9:])

# YY_MM_DD hh:mm:ss.ms  ->  hh:mm:ss.msmsms  ->  m_sec
def df_date2time(df_data, START_TIME):
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
