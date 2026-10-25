from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,precision_score,recall_score
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports';REPORT.mkdir(exist_ok=True)
ARTIFACT=ROOT/'artifacts';ARTIFACT.mkdir(exist_ok=True)
def dump(name,value):
 (REPORT/name).write_text(json.dumps(value,indent=2,default=lambda v:v.item())+'\n')
def logit(p):
 p=np.clip(np.asarray(p),1e-6,1-1e-6)
 return np.log(p/(1-p))
def fit_calibrator(model,x,y):
 c=LogisticRegression(C=1e6,solver='lbfgs',random_state=42)
 c.fit(logit(model.predict_proba(x)[:,1]).reshape(-1,1),y)
 return c
def predict(model,calibrator,x):
 return calibrator.predict_proba(logit(model.predict_proba(x)[:,1]).reshape(-1,1))[:,1]
def metrics(y,p):
 return {'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p),'brier':brier_score_loss(y,p),'prevalence':float(np.mean(y))}
def psi(reference,current,bins=10):
 edges=np.unique(np.quantile(reference,np.linspace(0,1,bins+1)))
 if len(edges)<2:return 0.0
 edges[0]=-np.inf;edges[-1]=np.inf
 a=np.histogram(reference,bins=edges)[0]+.5;b=np.histogram(current,bins=edges)[0]+.5
 a=a/a.sum();b=b/b.sum()
 return float(np.sum((b-a)*np.log(b/a)))
def chart(fig,name):
 fig.tight_layout();fig.savefig(REPORT/(name+'.svg'));fig.savefig(REPORT/(name+'.png'),dpi=130);plt.close(fig)
def interval_bootstrap(y,p,metric,repeats=200):
 rng=np.random.default_rng(42);y=np.asarray(y);p=np.asarray(p);values=[]
 for _ in range(repeats):
  i=rng.integers(0,len(y),len(y))
  if len(np.unique(y[i]))==2:values.append(metric(y[i],p[i]))
 return np.quantile(values,[.025,.975]).tolist()
