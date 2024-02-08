import requests

# The URL for the API endpoint (assuming the Flask app is running locally on port 5000)
url = "http://34.247.213.126:3000/api/biped_hq"

# Example data to send in the request
# Replace this with the actual RSSI signals data you want to send
data_to_send = {
    "tab": [
        [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ],  # Example RSSI signal value
        [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ],  # Another example RSSI signal value
        [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ],  # Add more RSSI signal values as needed
    ]
}

# Make the POST request to the biped_hq endpoint
response = requests.post(url, json=data_to_send)

# Check if the request was successful
if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Error:", response.status_code, response.text)
