import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from selenium.webdriver.common.by import By
url = "https://weather.com/weather/tenday/l/San+Francisco+CA?canonicalCityId=dfdaba8cbe3a4d12a8796e1f7b1ccc7174b4b0a2d5ddb1c8566ae9f154fa638c"
pth = 'chromedriver.exe'

driver = webdriver.Chrome(pth)
sleep(3)

driver.get(url)
sleep(3)
cookies = driver.find_element(By.XPATH, '/html/body/div[3]/div/div/div/div/div/div[3]/button[1]')
cookies.click()
sleep(3)