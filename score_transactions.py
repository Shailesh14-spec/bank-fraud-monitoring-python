"""Replay a local unlabelled transaction CSV into closed-hour analyst queues."""
from pathlib import Path
import argparse,joblib,numpy as np,pandas as pd
from banklab.common import predict,ROOT
from banklab.policy import review_queue
p=argparse.ArgumentParser();p.add_argument('input_csv',type=Path);p.add_argument('--capacity',type=int,default=10);p.add_argument('--output',type=Path,default=ROOT/'artifacts/replay_queue.csv');args=p.parse_args()
bundle=joblib.load(ROOT/'artifacts/model.joblib');raw=pd.read_csv(args.input_csv)
required=['Time','Amount']+[f'V{i}' for i in range(1,29)]
if not set(required).issubset(raw) or raw[required].isna().any().any() or (raw.Amount<0).any():raise ValueError('Missing/invalid transaction inputs')
x=raw[[f'V{i}' for i in range(1,29)]].copy();x['LogAmount']=np.log1p(raw.Amount)
frame=raw[['Time','Amount']].copy();frame['row_id']=np.arange(len(raw));frame['score']=predict(bundle['model'],bundle['calibrator'],x[bundle['features']])
queue=review_queue(frame,capacity=args.capacity)
args.output.parent.mkdir(parents=True,exist_ok=True);queue.to_csv(args.output,index=False)
print('Wrote',len(queue),'closed-hour review items to',args.output)
