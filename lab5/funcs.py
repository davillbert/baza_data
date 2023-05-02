import pandas as pd
import numpy as np

def hhmmss2ms(hms):
    '''
    hh:mm:ss.msmsms  ->  m_sec
    '''
    if '.' in hms and ':' in hms:
        res = (int(hms[:2]) * 60 * 60 + int(hms[3:5])*60 + int(hms[6:8]))*1000 + int(hms[9:])
    else:
        return hhmmss2ms(f'{hms}.000')
    return res 


def ms_to_time_str(ms):
    '''
    m_sec  ->  hh:mm:ss.msmsms
    '''
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def df_date_and_time_to_time_in_ms(df_data, t_col='Date and Time', START_TIME='00:00:00'):
    '''
    Убирает дату, а время переводит в мс.
    YY_MM_DD hh:mm:ss.msmsms  ->  hh:mm:ss.msmsms  ->  m_sec
    '''
    if '.' not in START_TIME and ':' in START_TIME:
        return df_date_and_time_to_time_in_ms(df_data, t_col=t_col, START_TIME=f'{START_TIME}.000')
    for i in range(len(df_data)):
        if ' ' not in str(START_TIME):
            df_data[t_col].at[i] = hhmmss2ms(df_data[t_col][i][9:]) - hhmmss2ms(START_TIME)
        else:
            df_data[t_col].at[i] = hhmmss2ms(df_data[t_col][i][9:]) - START_TIME

    return df_data


def cut_df(df, start, end, t_col='Date and Time', START_TIME='00:00:00'):
    '''
    Вырезает из таблицы пласт, ограниченный значениями start и end по значениям столбца t_col. 
    df, int, int  ->  df_cutted
    '''
    if t_col == 'Date and Time':
        res = df[(df[t_col] > (hhmmss2ms(start) - hhmmss2ms(START_TIME))) & (df[t_col] < (hhmmss2ms(end)- hhmmss2ms(START_TIME)))]
    else:
        res = df[(df[t_col] > (start)) & (df[t_col] < (end))]
    return res.reset_index().drop('index', axis=1)


def cut_df_alot(df, time_brackets, START_TIME='00:00:00', t_col='Date and Time'):
    '''
    Вырезает из таблицы пласты, ограниченные значениями из time_brackets по значениям столбца t_col. 
    Возвращает массив dataFrame'ов.
    
    df, [[int, int], ...]  ->  [df_cutted]
    '''
    res_array = []
    for i in range(len(time_brackets)):
        res_array.append(cut_df(df, time_brackets[i][0],  time_brackets[i][1], t_col=t_col, START_TIME=START_TIME))
    return res_array


def df_from_csv_w_cols(path, columns, delimiter=';'):
    '''
    Из csv файла path возвращает dataFrame со столбцами columns.
    '''
    df = pd.read_csv(path, header=0, usecols=columns, delimiter=';')
    
    return df


def get_vec_abs_from_components(df, xyz_cols, t_col='Date and Time', vec_name='Abs(Vector)'):
    '''
    Возвращает dataFrame с модулем вектора, который задаётся одной или несколькими компонентами.
    Названия столбцов компонент в df передаётся в виде массива xyz_cols.

    dataFrame, [x_1,...,x_n]  ->  dataFrame:[Time, Vec]
    '''
    squares_sum = np.zeros(len(df[xyz_cols[0]]))
    for i in range(len(xyz_cols)):
        for j in range(len(df[xyz_cols[i]])):
            squares_sum[j] += (df[xyz_cols[i]][j]) ** 2
    for i in range(len(squares_sum)):
        squares_sum[i] = np.sqrt(squares_sum[i])

    vec_df = pd.DataFrame(squares_sum, columns=[vec_name])
    
    new_df = pd.concat([df[t_col], vec_df], axis=1, ignore_index=True)
    new_df.columns = [t_col, vec_name]
    return new_df


def get_std_of_vec(df, vec_name):
    '''
    Возвращает СКО/СР.ЗНАЧ от всех значений столбца vec_name в df.
    df, str  ->  float
    '''
    arr = np.array(df[vec_name])
    std = arr.std()/arr.mean()
    return std


def get_ampl_of_vec(df, vec_name):
    '''
    Возвращает амплитуду (max-min / 2) от всех значений столбца vec_name в df.
    df, str  ->  float
    '''
    arr = np.array(df[vec_name])
    return (arr.max() - arr.min()) / 2