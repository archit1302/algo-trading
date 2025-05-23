import urllib.parse
import pandas as pd
import requests
from Config import API_KEY, SECRET_KEY, RURL

apiKey = API_KEY
secretKey = SECRET_KEY
redirectUrl = RURL
rurl = urllib.parse.quote(redirectUrl,safe="")


uri = f'https://api-v2.upstox.com/login/authorization/dialog?response_type=code&client_id={apiKey}&redirect_uri={rurl}'
print(uri)



