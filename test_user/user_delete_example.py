# delete_model_user.py

import requests

# Name of the model to delete (must match what was returned in upload)
# You can pass this dynamically or hardcode based on previous upload response
model_name = input("Enter model filename to delete (e.g., user_model.pkl): ").strip()

# Send DELETE request
response = requests.delete(f"http://127.0.0.1:8000/delete_model/{model_name}")

# Print results
print("Status Code:", response.status_code)
try:
    print("Response:", response.json())
except Exception:
    print("Raw Text:", response.text)
