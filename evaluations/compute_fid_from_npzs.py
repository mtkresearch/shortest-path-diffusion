import sys
from evaluations.fid_score import compute_fid

reference_batch = sys.argv[1]
sample_batch = sys.argv[2]

print(f'Evaluating {sample_batch}')
fid = compute_fid(sample_batch, reference_batch, batch_size=100, cuda_device_id=0)
print(f'FID: {fid:.2f}')