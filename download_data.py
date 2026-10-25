from pathlib import Path
from urllib.request import urlopen
import hashlib
ROOT=Path(__file__).resolve().parent
URL='https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
EXPECTED='76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89'
(ROOT/'data').mkdir(exist_ok=True);path=ROOT/'data/creditcard.csv';temp=path.with_suffix('.download')
with urlopen(URL,timeout=120) as response,temp.open('wb') as f:
 while True:
  block=response.read(1024*1024)
  if not block:break
  f.write(block)
if hashlib.sha256(temp.read_bytes()).hexdigest()!=EXPECTED:
 temp.unlink();raise ValueError('Source checksum changed; review upstream before updating')
temp.replace(path)
print('Verified public card-fraud CSV. Original attribution and data terms: docs/DATA_SOURCE.md')
