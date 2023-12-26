from test import processing, select_columns_RFC

if __name__ == '__main__':
    df_concat = processing()
    select_columns_RFC(df_concat, cnt=1000)

