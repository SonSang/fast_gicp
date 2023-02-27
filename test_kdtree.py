import pygicp
import torch as th
import time

size = 8192
device = 'cpu'

pc = th.rand((size, 3), dtype=th.float32, device=device)

start_time = time.time()
kdtree = pygicp.PCLKDTree()
kdtree.set_input_cloud(pc.cpu().numpy())
end_time = time.time()

print("Build Time: {:.2f} ms".format((end_time - start_time) * 1e3))

for i in range(15):
    curr_size = 2 ** i
    curr_pc = th.rand((curr_size, 3), dtype=th.float32, device=device)
    
    start_time = time.time()
    kdtree.knn(curr_pc.cpu().numpy(), 20, 1)
    end_time = time.time()
    
    print("KNN # Point: {} / Time: {:.2f} ms".format(curr_size, (end_time - start_time) * 1e3))
    