import pandas as pd
import glob
import os
from datetime import datetime

path = os.getcwd()#my folder path with csv files
file_grex = os.path.join(path, "csv/*.csv")
csv_files = glob.glob(file_grex)


appropriate_columns = ['Timestamp', 'Longitude', 'Latitude', 'NetworkTech', 'NetworkMode', 'Level', 'Qual', 'SNR', 'CQI', 'LTERSSI', 'DL_bitrate', 'UL_bitrate', 'Altitude', 'Height', 'State', 'EVENT', 'Eid']
#for f in csv_files:
#    df = pd.read_csv(f)
    # do a set difference of column names and check if length is 0
    #if len(set(df.columns) - set(appropriate_columns)) != 0 or len(set(appropriate_columns) - set(df.columns)) != 0:
    #    if '5G' in set(df["NetworkTech"]):
    #        print(f'Inappropriate Columns exist in file with tech: {set(df["NetworkTech"])} of file: {f}')



dataframes = []
keep_columns = ['Timestamp', 'DL_bitrate', 'UL_bitrate', 'Filename']

for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['Filename'] = csv_file.split('/')[-1]
            df = df[df['NetworkTech'] == '5G']
            df = df[keep_columns]
            dataframes.append(df)
        except Exception as e:
             print(f'File: {csv_file}')
        

combined_df = pd.concat(dataframes, ignore_index=True)
combined_df = combined_df[combined_df['DL_bitrate'].astype(float) > 100]

date_format = "%Y.%m.%d_%H.%M.%S"

combined_df['Timestamp'] = combined_df['Timestamp'].apply(lambda x: datetime.strptime(x, date_format))

combined_df.to_csv('output.csv', index=False)