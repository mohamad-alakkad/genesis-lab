import numpy as np, sys
blocks=[]
try:
    for i in range(25):  # 25 x 100 MB = 2.5 GB, touched
        a=np.ones(100*1024*1024//8); blocks.append(a)
    print("ALLOCATED_2.5GB"); sys.exit(0)
except MemoryError:
    print(f"MEMORYERROR_AT_MB={len(blocks)*100}"); sys.exit(3)
