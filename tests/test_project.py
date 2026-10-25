from pathlib import Path
import json,unittest
import numpy as np
import pandas as pd
from banklab.policy import review_queue,queue_metrics
ROOT=Path(__file__).resolve().parents[1]
class PolicyTests(unittest.TestCase):
 def test_strict_capacity_and_tie_break(self):
  d=pd.DataFrame({'row_id':[4,2,3,1,5,6],'Time':[1,2,3,4,3601,3602],'score':[.9,.9,.1,.8,.2,.3]})
  q=review_queue(d,capacity=1);self.assertEqual(q.row_id.tolist(),[2,6]);self.assertTrue(q.groupby('hour').size().le(1).all())
  self.assertEqual(len(review_queue(d,capacity=0)),0)
  with self.assertRaises(ValueError):review_queue(d,capacity=-1)
 def test_recall_and_amount_denominators(self):
  d=pd.DataFrame({'Class':[1,1,0],'Amount':[10,30,100]});q=d.iloc[[0,2]];m=queue_metrics(d,q)
  self.assertEqual(m['recall'],.5);self.assertEqual(m['precision'],.5);self.assertEqual(m['fraud_amount_capture_share'],.25)
class CompletedRunTests(unittest.TestCase):
 def test_time_boundaries_and_gap(self):
  d=pd.read_csv(ROOT/'reports/split_summary.csv').set_index('split')
  for a,b in [('train','calibration'),('calibration','validation'),('validation','test')]:self.assertGreaterEqual(d.loc[b,'min_time']-d.loc[a,'max_time'],3600)
 def test_row_accounting(self):
  m=json.loads((ROOT/'reports/metrics.json').read_text());self.assertEqual(m['source_rows'],m['unique_event_rows']+m['duplicates_removed']);self.assertEqual(sum(m['split_rows'].values())+m['purged_gap_rows'],m['unique_event_rows'])
 def test_budget_and_summary_reconciliation(self):
  p=pd.read_csv(ROOT/'artifacts/test_predictions.csv');q=pd.read_csv(ROOT/'artifacts/review_queue.csv');m=json.loads((ROOT/'reports/metrics.json').read_text())
  self.assertTrue(q.groupby('hour').size().le(10).all());self.assertTrue(set(q.row_id).issubset(set(p.row_id)))
  self.assertEqual(int(q.Class.sum()),m['queue_10_per_hour']['flagged_fraud']);self.assertEqual(len(q),m['queue_10_per_hour']['reviews']);self.assertTrue(p.score.between(0,1).all())
 def test_capacity_curve_monotonic(self):
  d=pd.read_csv(ROOT/'reports/review_capacity.csv');d=d.loc[d.policy=='model'].sort_values('capacity_per_hour')
  self.assertTrue(d.reviews.is_monotonic_increasing);self.assertTrue(d.flagged_fraud.is_monotonic_increasing)
if __name__=='__main__':unittest.main()
