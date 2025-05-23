import urllib.parse
import pandas as pd
import requests


access_token = ""

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