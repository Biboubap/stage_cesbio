from samples_set import SamplesSet
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # #POP LICHEN 1
    # x_1 = 12040
    # y_1 = 15730
    
    # row_start = int(np.round(x_1/32))*32 #row X
    # column_start = int(np.round(y_1/32))*32 #column Y
    # n_samples_x = 20
    # n_samples_y = 20
    
    # size_patch= 32
    # pop_lichen_1 = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y, category="lichen")
    # pop_lichen_1.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)
    # pop_lichen_1.plot_samples()

    # path = "data/samples/lichen/"
    # pop_lichen_1.save_samples_to_json(filename=path+"pop_lichen_1.json")


    # #POP LICHEN 2
    # x_2 = 19300
    # y_2 = 8520
    
    # row_start = int(np.round(x_2/32))*32 #row X
    # column_start = int(np.round(y_2/32))*32 #column Y
    # n_samples_x = 20
    # n_samples_y = 20
    # size_patch= 32
    # pop_lichen_2 = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y, category="lichen")
    # pop_lichen_2.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)
    # pop_lichen_2.plot_samples()
    # path = "data/samples/lichen/"
    # pop_lichen_2.save_samples_to_json(filename=path+"pop_lichen_2.json")


    #POP SPHEGNES 1
    x_s1 = 16134
    y_s1 = 8910
    n_samples_x = 30
    n_samples_y = 30
    size_patch= 32
    
    row_start = int(np.round(x_s1/32))*32 #row X
    column_start = int(np.round(y_s1/32))*32 #column Y
    pop_sphegnes_1 = SamplesSet(n_samples_x=n_samples_x, n_samples_y=n_samples_y, category="sphegnes")
    pop_sphegnes_1.create_samples_grid(x_start=row_start, y_start=column_start, size_patch=size_patch)
    pop_sphegnes_1.plot_samples()
    path = "data/samples/sphegnes/"
    pop_sphegnes_1.save_samples_to_json(filename=path+"pop_sphegnes_1.json")

    