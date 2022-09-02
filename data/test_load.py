import io
from petrel_client.client import Client
conf_path = '~/petreloss.conf'
client = Client(conf_path)
import logging
LOG = logging.getLogger('petrel_client.test')
import numpy as np

f_url = 's3://caiyingjie/1462.txt'
body = client.get(f_url, update_cache=True)
f = io.BytesIO(body)
c2w = np.loadtxt(f).astype(np.float32)
print (c2w)
#data = np.load(f)
