import pandas as pd
import os

directory = 'C:\\Users\\SUBHAYAN\\Downloads\\Sample AI agent\\Excels' 
filename = 'my_data.xlsx'
full_path = os.path.join(directory, filename)

df = pd.read_excel(full_path)
filtered_df = df[df["Application Name"].str.contains("lyric", case=False, na=False)]
server_list = filtered_df.to_dict(orient="records")
print(server_list)