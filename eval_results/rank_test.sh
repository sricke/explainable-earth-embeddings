cd /home/libe2152/projects/explainable-earth-embeddings/eval_results && python3 -c "
import sys; sys.path.append('..')
import torch
from splice import _LocWrapper
from paths import load_paths
p = load_paths()
for name, key in [('satclip','satclip_test_2000'),('climplicit','climplicit_test_2000')]:
    m = torch.load(p['splice'][key]['model'], map_location='cpu', weights_only=False)
    C = m.dictionary
    print(name, 'dict', tuple(C.shape), 'l1', m.l1_penalty, 'solver', m.solver)
    S = torch.linalg.svdvals(C)
    rank = (S > 1e-3).sum().item()
    print(' rank', rank, '/', C.shape[1], ' cond', round((S.max()/S.min()).item(), 2))
" 2>&1 | grep -v "NVML\|UserWarning\|raw_cnt"