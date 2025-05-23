
from playwright.sync_api import Playwright, sync_playwright
from urllib.parse import parse_qs,urlparse,quote
import pyotp
import requests
import Config


rurlEncode = quote(Config.RURL,safe="")

AUTH_URL = f'https://api-v2.upstox.com/login/authorization/dialog?response_type=code&client_id={Config.API_KEY}&redirect_uri={rurlEncode}'
def getAccessToken(code):
    url = 'https://api-v2.upstox.com/login/authorization/token'

    headers = {
        'accept': 'application/json',
        'Api-Version': '2.0',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'code': code,
        'client_id': Config.API_KEY,
        'client_secret': Config.SECRET_KEY,
        'redirect_uri': Config.RURL,
        'grant_type': 'authorization_code'
    }

    response = requests.post(url, headers=headers, data=data)
    json_response = response.json()

    # Access the response data
    # print(f"access_token:  {json_response['access_token']}")
    return json_response['access_token']

def run(playwright: Playwright) -> str:
    browser = playwright.firefox.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()   
    with page.expect_request(f"*{Config.RURL}?code*") as request:
        page.goto(AUTH_URL)
        page.locator("#mobileNum").click()
        page.locator("#mobileNum").fill(Config.MOBILE_NO)
        page.get_by_role("button", name="Get OTP").click()
        page.locator("#otpNum").click()
        otp = pyotp.TOTP(Config.TOTP_KEY).now()
        page.locator("#otpNum").fill(otp)
        page.get_by_role("button", name="Continue").click()
        page.get_by_label("Enter 6-digit PIN").click()
        page.get_by_label("Enter 6-digit PIN").fill(Config.PIN)
        res = page.get_by_role("button", name="Continue").click()
        page.wait_for_load_state()

    url =    request.value.url 
    print(f"Redirect Url with code : {url}")
    parsed = urlparse(url)
    code = parse_qs(parsed.query)['code'][0]
    context.close()
    browser.close()
    return code



def getToken():
    with sync_playwright() as playwright:
        code = run(playwright)
    token = getAccessToken(code)
    return token

access_token = getToken()

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