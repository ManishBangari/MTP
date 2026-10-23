import numpy as np

# Use a context manager to open the file safely and avoid leaking resources
with np.load('../results/raw/fig3.npz') as data:
    # 1. Check what arrays are inside the file
    print("Array keys:", data.files)
    
    # 2. Extract an array by its key name
    my_array = data['fig3_g0_DAQ_F_s3']
    print(my_array)
