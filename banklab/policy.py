"""Closed-hour batch review policy: never exceeds the configured analyst capacity."""
import numpy as np
import pandas as pd
def review_queue(frame,score_column='score',capacity=10):
 if capacity<0:raise ValueError('Capacity cannot be negative')
 required={'row_id','Time',score_column}
 if not required.issubset(frame):raise ValueError('Missing queue fields')
 d=frame.copy();d['hour']=(d.Time//3600).astype(int)
 ranked=d.sort_values(['hour',score_column,'row_id'],ascending=[True,False,True],kind='stable')
 queue=ranked.groupby('hour',sort=True).head(capacity).copy()
 queue['rank_in_hour']=queue.groupby('hour').cumcount()+1
 return queue
def queue_metrics(frame,queue):
 positives=int(frame.Class.sum());hits=int(queue.Class.sum());amount=float(frame.loc[frame.Class==1,'Amount'].sum())
 return {'reviews':len(queue),'flagged_fraud':hits,'total_fraud':positives,'precision':hits/len(queue) if len(queue) else 0.0,'recall':hits/positives if positives else 0.0,'fraud_amount_capture_share':float(queue.loc[queue.Class==1,'Amount'].sum())/amount if amount else 0.0}
