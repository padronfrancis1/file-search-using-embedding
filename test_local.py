import requests
import json

url = "http://127.0.0.1:8000/ask_localai"

payload = json.dumps({
  "query": "Is there a case number for OWASP?"
})
headers = {
  'Accept': 'application/json',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
print("-------------------")
print(response)