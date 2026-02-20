import splitfolders

#Input folder (the one with 0, 1, 2... 100 subfolders)
input_folder = "images" 

#Output folder where 'train', 'val', and 'test' will be created
output_folder = "data_split"

#Split with a ratio of 70% Train, 15% Val, 15% Test
#seed=42 ensures the split is reproducible (same every time you run it)
splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=42, 
    ratio=(0.7, 0.15, 0.15), 
    group_prefix=None, 
    move=False # Set to True if you want to move files instead of copying them
)

print(f"Data split successfull: Check the '{output_folder}' directory.")
