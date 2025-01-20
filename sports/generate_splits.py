import os
import pandas as pd
import random
import sys

def split(src_dir, csv_dir, window, slide):
    # Process each CSV file in the source folder
    for filename in os.listdir(src_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(src_dir, filename)
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Find the index of the first row where `label` is 0
            zero_label_index = df[df['label'] == 0].index.min()
            
            # Check if a row with `label` == 0 was found
            if zero_label_index is not None:
                # Randomly select an integer f between 10 and 20
                f = random.randint(10, 20)
                
                # Calculate the start index for keeping rows
                start_index = max(zero_label_index - f, 0)  # Ensure start index is non-negative
                
                # If there are at least 10 rows before `label` == 0, trim rows before the start index
                if zero_label_index >= 10:
                    df = df.iloc[start_index:]
            
            # Initialize starting position for sliding windows
            start_pos = 0
            window_number = 1

            # Divide the DataFrame into multiple overlapping windows
            while start_pos + window <= len(df):
                # Extract a window of rows
                window_df = df.iloc[start_pos:start_pos + window]

                # Define the output file path for this window
                output_filename = f"{filename.rstrip('.csv')}_window{window_number}.csv"
                output_file_path = os.path.join(csv_dir, output_filename)

                # Save the window as a separate CSV file
                window_df.to_csv(output_file_path, index=False)
                
                # Increment the window count and slide the start position
                window_number += 1
                start_pos += (window - slide)

            if start_pos < len(df):
                # Calculate how many rows are needed to fill the window
                needed_rows = window - (len(df) - start_pos)
                
                # Create a new DataFrame with the last row replicated if needed
                last_row = df.iloc[-1]
                if int(last_row['label']) != 8: # if the last row is not a no-event frame, don't continue 
                    print(f'bug! event {int(last_row["label"])} will be over-represented')
                    sys.exit(1)
                
                window_df = pd.concat([df.iloc[start_pos:], pd.DataFrame([last_row] * needed_rows)])

                # Define the output file path for the last overlapping window
                output_filename = f"{filename.rstrip('.csv')}_window{window_number}.csv"
                output_file_path = os.path.join(csv_dir, output_filename)
                
                # Save the last window as a separate CSV file
                window_df.to_csv(output_file_path, index=False)