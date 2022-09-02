import numpy as np
data = np.load('rank0_seqid0_step40.npz')
print(data.files)
print(data['xyz'].shape, data['embed'].shape)
exit()
