original data has two folders corresponding to test and train videos.
each csv has pose data for one frame per row. no other modifications

we run trim_csv.py to trim the csvs in the original_data to remove frames before address and after finish
this will generate src_data folder

then run train.py it will create once csv per one window (only for the training data) in the splits folder.
the last window is padded by the data of the last frame to complete the window size

