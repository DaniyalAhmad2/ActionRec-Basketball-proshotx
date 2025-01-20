import pandas as pd

# Read the original CSV file
df = pd.read_csv('Front-60fps-events.csv')

# Split the DataFrame based on the lstm_category column
df_shot_phase = df[df['lstm_category'] == 'Shot_Phase']
df_shot_status = df[df['lstm_category'] == 'Shot_Status']

# Write the DataFrames to two separate CSV files
df_shot_phase.to_csv('shot_phase.csv', index=False)
df_shot_status.to_csv('shot_status.csv', index=False)

print("CSV files created: shot_phase.csv and shot_status.csv")