import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, time, torch, warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(4)
from src.trials_adapter import get_trials
from src.pipeline_phase4 import (build_raw_trials, windows_from_signal,
                                 infer_f1, REF_CHANNEL)
from src.pipeline_phase3 import GaitCNN, GaitLSTM
from src.perturbations_mc import (apply_perturbation_mc, normalize,
                                  PERTURBATIONS_MC)
MAP={'CNN':GaitCNN,'LSTM':GaitLSTM}
pos, model_name = sys.argv[1], sys.argv[2]
SUBJ = sys.argv[3].split(',') if len(sys.argv)>3 else None
ref=REF_CHANNEL[pos]
trials=get_trials(verbose=False)
raw=build_raw_trials(trials,pos)
conds=[('baseline',0)]+[(p,l) for p,i in PERTURBATIONS_MC.items() for l in i['levels']]
rows=[]; t0=time.time()
for subj in sorted({r['subject'] for r in raw}):
    if SUBJ and subj not in SUBJ: continue
    m=MAP[model_name](9,3)
    m.load_state_dict(torch.load(f'results/models/{model_name}_{pos}_{subj}.pt',
                                 map_location='cpu')); m.eval()
    for rt in [r for r in raw if r['subject']==subj]:
        for scope in ['all9','gyro3','ref1']:
            for norm in ['adaptive','frozen']:
                for pt,lv in conds:
                    if pt=='baseline' and scope!='all9': continue
                    sp=apply_perturbation_mc(pt,rt['sig9'],rt['t'],lv,ref,scope)
                    X,_,tc=windows_from_signal(normalize(sp,norm,rt['clean_stats']),
                                               rt['t'],rt['gt_HS'],rt['gt_TO'],False)
                    if X is None: continue
                    fH,fT,nH,nT=infer_f1(m,X,tc,rt['gt_HS'],rt['gt_TO'])
                    rows.append(dict(model=model_name,position=pos,subject=subj,
                        gender=rt['gender'],speed=rt['speed'],rep=rt['rep'],
                        trial_id=rt['trial_id'],scope=scope,norm=norm,
                        pert_type=pt,pert_level=lv,F1_HS=fH,F1_TO=fT,
                        n_det_HS=nH,n_det_TO=nT))
        print(f'  {rt["trial_id"]:<20} {len(rows):>6}행 {time.time()-t0:.0f}s',flush=True)
tag='' if not SUBJ else '_'+'-'.join(SUBJ)
pd.DataFrame(rows).to_csv(f'results/dl_p4_{model_name}_{pos}{tag}.csv',index=False)
print(f'[저장] dl_p4_{model_name}_{pos}{tag}.csv  {len(rows)}행  {time.time()-t0:.0f}s')