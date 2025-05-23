import urllib.parse
import pandas as pd
import requests
from Config import API_KEY, SECRET_KEY, RURL

apiKey = API_KEY
secretKey = SECRET_KEY
redirectUrl = RURL
rurl = urllib.parse.quote(redirectUrl,safe="")


url = 'https://api-v2.upstox.com/login/authorization/token'
code = 'oaqqvQ'
headers = {
    'accept': 'application/json',
    'Api-Version': '2.0',
    'Content-Type': 'application/x-www-form-urlencoded'
}

data = {
    'code': code,
    'client_id': apiKey,
    'client_secret': secretKey,
    'redirect_uri': redirectUrl,
    'grant_type': 'authorization_code'
}

response = requests.post(url, headers=headers, data=data)
json_response = response.json()

access_token = json_response['access_token']
# print(access_token)


url = 'https://api-v2.upstox.com/user/get-funds-and-margin'
headers = {
    'accept': 'application/json',
    'Api-Version': '2.0',
    'Authorization': f'Bearer {access_token}'
}
params = {
    'segment': 'COM'  #'COM'
}

response = requests.get(url, headers=headers, params=params)
print(response.json())