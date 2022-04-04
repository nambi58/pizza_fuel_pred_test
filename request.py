import requests

url = 'http://localhost:5000/predict_api' 
r = requests.post(url,json={ 'age':20,'wieght':45,'distance':200})

print(r.json())


