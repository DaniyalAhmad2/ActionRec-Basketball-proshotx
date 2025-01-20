import os
import pandas as pd
import random
import sys

def trim(src_dir, csv_dir):
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
            
            label_7_index = df[df['label'] == 7].index.max()

            # Check if a row with `label` == 7 was found
            if label_7_index is not None:
                # Randomly select an integer f between 10 and 20
                f = random.randint(10, 20)
                
                # Calculate the end index for keeping rows
                end_index = min(label_7_index + f, len(df))  # Ensure end index is within DataFrame length
                
                # If there are at least 10 rows after `label` == 7, trim rows after the end index
                if len(df) - label_7_index >= 10:
                    df = df.iloc[:end_index]

                
            # Save the trimmed DataFrame as a separate CSV file
            output_file_path = os.path.join(csv_dir, filename)
            df.to_csv(output_file_path, index=False)


if __name__ == '__main__':

    src_dir = 'original_data/30fps_test'
    csv_dir = 'src_data/30fps_test'
    os.makedirs(csv_dir, exist_ok=True)
    trim(src_dir, csv_dir)

    src_dir = 'original_data/30fps_train'
    csv_dir = 'src_data/30fps_train'
    os.makedirs(csv_dir, exist_ok=True)
    trim(src_dir, csv_dir)
