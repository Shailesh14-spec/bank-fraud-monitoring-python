"""Chronological card-fraud triage with separate calibration and capacity controls."""
from pathlib import Path
import argparse,hashlib,json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,precision_recall_curve
from sklearn.inspection import permutation_importance
from banklab.common import *
from banklab.policy import review_queue,queue_metrics

def main(source):
 raw=pd.read_csv(source);assert raw.shape==(284807,31)
 assert not raw.isna().any().any() and set(raw.Class.unique())=={0,1} and (raw.Amount>=0).all()
 columns=['Time']+[f'V{i}' for i in range(1,29)]+['Amount']
 # Remove exact repeated event profiles, independent of the outcome label.
 conflicts=raw.groupby(columns,dropna=False).Class.nunique().gt(1).sum();assert conflicts==0
 d=raw.drop_duplicates(columns,keep='first').copy();d['row_id']=d.index;d=d.sort_values(['Time','row_id']).reset_index(drop=True)
 masks={'train':d.Time<82800,'calibration':d.Time.between(86400,104400,inclusive='left'),'validation':d.Time.between(108000,126000,inclusive='left'),'test':d.Time>=129600}
 tr,ca,va,te=[masks[k] for k in ['train','calibration','validation','test']]
 assert d.loc[tr,'Time'].max()+3600<=d.loc[ca,'Time'].min()
 assert d.loc[ca,'Time'].max()+3600<=d.loc[va,'Time'].min()
 assert d.loc[va,'Time'].max()+3600<=d.loc[te,'Time'].min()
 x=d[[f'V{i}' for i in range(1,29)]].copy();x['LogAmount']=np.log1p(d.Amount)
 y=d.Class.astype(int)
 candidates={'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42)), 'gradient_boosting':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=20,class_weight='balanced',early_stopping=False,random_state=42)}
 fitted={};comparison=[]
 for name,model in candidates.items():
  model.fit(x.loc[tr],y[tr]);cal=fit_calibrator(model,x.loc[ca],y[ca]);p=predict(model,cal,x.loc[va])
  frame=d.loc[va,['row_id','Time','Class','Amount']].copy();frame['score']=p;queue=review_queue(frame,capacity=10)
  comparison.append({'model':name,**metrics(y[va],p),'validation_recall_10_per_hour':queue_metrics(frame,queue)['recall'],'calibration_slope':float(cal.coef_[0,0])});fitted[name]=(model,cal)
 comp=pd.DataFrame(comparison).sort_values(['validation_recall_10_per_hour','average_precision'],ascending=False);selected=comp.iloc[0].model;model,cal=fitted[selected]
 pv=predict(model,cal,x.loc[va]);threshold=float(np.quantile(pv,.995))
 p=predict(model,cal,x.loc[te]);frame=d.loc[te,['row_id','Time','Class','Amount']].copy();frame['score']=p
 policies=[];queues={}
 for cap in [5,10,20,40]:
  q=review_queue(frame,capacity=cap);queues[cap]=q;policies.append({'policy':'model','capacity_per_hour':cap,**queue_metrics(frame,q)})
 for cap in [10]:
  q=review_queue(frame,score_column='Amount',capacity=cap);policies.append({'policy':'largest_amount_baseline','capacity_per_hour':cap,**queue_metrics(frame,q)})
 policies=pd.DataFrame(policies);policies.to_csv(REPORT/'review_capacity.csv',index=False)
 selected_queue=queues[10];qm=queue_metrics(frame,selected_queue)
 # Hour-block bootstrap: limited to observed hours, not unseen cards or days.
 rng=np.random.default_rng(42);hour=(frame.Time//3600).astype(int).to_numpy();blocks=[np.flatnonzero(hour==v) for v in np.unique(hour)];aps=[]
 actual=frame.Class.to_numpy()
 for _ in range(250):
  idx=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
  if len(np.unique(actual[idx]))==2:aps.append(average_precision_score(actual[idx],p[idx]))
 result={'source_rows':len(raw),'unique_event_rows':len(d),'duplicates_removed':len(raw)-len(d),'purged_gap_rows':int((~(tr|ca|va|te)).sum()),'split_rows':{k:int(v.sum()) for k,v in masks.items()},'split_frauds':{k:int(y[v].sum()) for k,v in masks.items()},'selected_model':selected,'test':metrics(actual,p),'test_average_precision_hour_block_95_ci':np.quantile(aps,[.025,.975]).tolist(),'queue_10_per_hour':qm,'validation_99_5pct_threshold':threshold,'test_threshold_review_share':float((p>=threshold).mean()),'test_threshold_recall':recall_score(actual,p>=threshold),'score_psi_train_test':psi(predict(model,cal,x.loc[tr]),p),'raw_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
 comp.to_csv(REPORT/'validation_comparison.csv',index=False)
 hourly=frame.assign(hour=hour).groupby('hour').agg(transactions=('Class','size'),frauds=('Class','sum'),mean_score=('score','mean'))
 queued=selected_queue.groupby('hour').agg(reviews=('Class','size'),flagged_fraud=('Class','sum'))
 hourly=hourly.join(queued).fillna(0);hourly.to_csv(REPORT/'hourly_monitoring.csv')
 # Explaining anonymised PCA components cannot produce customer-level business reasons.
 importance=permutation_importance(model,x.loc[va],y[va],scoring='average_precision',n_repeats=2,random_state=42,n_jobs=2)
 pd.DataFrame({'feature':x.columns,'validation_ap_drop':importance.importances_mean}).sort_values('validation_ap_drop',ascending=False).to_csv(REPORT/'validation_importance.csv',index=False)
 dump('metrics.json',result)
 dump('data_audit.json',{'missing_cells':int(raw.isna().sum().sum()),'conflicting_event_labels':int(conflicts),'model_features':x.columns.tolist(),'time_used_only_for_splitting_and_queue':True,'gap_seconds':3600,'window_bounds':{k:[float(d.loc[v,'Time'].min()),float(d.loc[v,'Time'].max())] for k,v in masks.items()}})
 pd.DataFrame([{'split':name,'rows':int(mask.sum()),'frauds':int(y[mask].sum()),'prevalence':float(y[mask].mean()),'min_time':float(d.loc[mask,'Time'].min()),'max_time':float(d.loc[mask,'Time'].max())} for name,mask in masks.items()]).to_csv(REPORT/'split_summary.csv',index=False)
 frame.to_csv(ARTIFACT/'test_predictions.csv',index=False);selected_queue.to_csv(ARTIFACT/'review_queue.csv',index=False)
 joblib.dump({'model':model,'calibrator':cal,'threshold':threshold,'features':x.columns.tolist()},ARTIFACT/'model.joblib')
 fig,axs=plt.subplots(2,2,figsize=(13,9));fig.suptitle('Card Fraud: Chronological Review-Capacity Benchmark',fontsize=18,weight='bold')
 pr,re,_=precision_recall_curve(actual,p);axs[0,0].plot(re,pr,color='#167D8D');axs[0,0].axhline(actual.mean(),ls='--',color='grey');axs[0,0].set(title='Held-out precision–recall',xlabel='Recall',ylabel='Precision')
 modelpol=policies.loc[policies.policy=='model'];axs[0,1].plot(modelpol.reviews,modelpol.recall,'o-',color='#4275B5');axs[0,1].set(title='Analyst capacity vs fraud capture',xlabel='Total reviews over test hours',ylabel='Recall')
 hourly[['frauds','flagged_fraud']].rename(columns={'frauds':'Labelled fraud','flagged_fraud':'Flagged fraud'}).plot.bar(ax=axs[1,0],color=['#E6A23C','#167D8D']);axs[1,0].set(title='True and flagged fraud by held-out hour',xlabel='Hour since first transaction',ylabel='Transactions')
 axs[1,1].bar(['Model queue','Largest amount'],[qm['recall'],float(policies.loc[policies.policy=='largest_amount_baseline','recall'].iloc[0])],color=['#167D8D','#C86464']);axs[1,1].set(title='Equal-capacity policy comparison',ylabel='Fraud recall');chart(fig,'monitoring-dashboard')
 (REPORT/'FINDINGS.md').write_text(f'''# Measured findings\n\nValidation selected **{selected}**. On the later held-out period, average precision is **{result['test']['average_precision']:.3f}** against prevalence **{result['test']['prevalence']:.4f}**; ROC AUC is {result['test']['roc_auc']:.3f}.\n\nAt **10 reviews per closed hour**, the model flags **{qm['flagged_fraud']} of {qm['total_fraud']}** labelled fraud transactions ({qm['recall']:.1%} recall) with {qm['precision']:.1%} precision and {qm['reviews']} total reviews. Flagged fraud amount share is {qm['fraud_amount_capture_share']:.1%}; this is not a prevented-loss estimate. The source amount's currency is not assumed in this project.\n\nA validation-selected probability threshold yields {result['test_threshold_review_share']:.2%} test review share, which need not respect an hourly budget. The strict closed-hour queue and fixed-threshold policy are separate operational designs.\n\nRecommendation: treat this as a reproducible offline card-fraud triage benchmark. Data covers about two days, PCA feature semantics are unavailable, and realistic label delays are not represented. A one-hour gap prevents adjacent windows touching but does not simulate delayed chargebacks. No AML detection, real-time system, customer-level explanation or deployment performance is claimed.\n''')
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'data/creditcard.csv');main(parser.parse_args().source)
