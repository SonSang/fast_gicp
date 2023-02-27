import numpy as np
import pygicp

pc = np.random.random((8192, 3)) * 10.

downsample = pygicp.downsample(pc, 0.5)
downsample_1 = pygicp.downsample_voxel_avg(pc, 0.5)
downsample_2 = pygicp.downsample_voxel_cen(pc, 0.5)

print(downsample.shape[0])
print(downsample_1.shape[0])
print(downsample_2.shape[0])