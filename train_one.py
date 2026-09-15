import sys; sys.path.insert(0,'.')
import time, torch
torch.set_num_threads(4)
from src.trials_adapter import get_trials
from src.pipeline_phase4 import build_raw_trials, get_or_train_model
from src.pipeline_phase3 import GaitCNN, GaitLSTM
MAP={'CNN':GaitCNN,'LSTM':GaitLSTM}
name,pos = sys.argv[1], sys.argv[2]
subs = sys.argv[3].split(',')
trials = get_trials(verbose=False)
raw = build_raw_trials(trials,pos)
for s in subs:
    t0=time.time(); print(f'=== {name} {pos} test={s} ===',flush=True)
    get_or_train_model(name,MAP[name],pos,s,raw)
    print(f'    {time.time()-t0:.0f}s',flush=True)