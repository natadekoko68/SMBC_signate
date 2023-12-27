from preprocessing.arrangement import processing
from estimate.estimator import RFC,output

if __name__ == "__main__":
    df_concat = processing()
    RFC(df_concat)
    output(df_concat)
