import requests
from app.utils.utils import time_and_log


@time_and_log
def main():
    # The URL for the API endpoint (assuming the Flask app is running locally on port 3000)
    url = "http://34.247.213.126:3000/mairie-xx/cluster"
    # url = "http://127.0.0.1:3000/mairie-xx/cluster"

    # Example data to send in the request
    # Replace this with the actual RSSI signals data you want to send
    # Biped requires 12 RSSI values per sample
    # This corresponds to a cluster 2 prediction
    data_to_send = {
        "tab": [
            [
                -82.0,
                -68.0,
                -80.0,
                -68.0,
                -83.0,
                -81.0,
                -88.0,
                -100.0,
                -100.0,
                -100.0,
                -100.0,
                -100.0,
            ],
            [
                -85.0,
                -76.0,
                -84.0,
                -63.0,
                -82.0,
                -80.0,
                -100.0,
                -100.0,
                -100.0,
                -100.0,
                -100.0,
                -100.0,
            ],
            [
                -90.0,
                -75.0,
                -90.0,
                -79.0,
                -82.0,
                -73.0,
                -92.0,
                -100.0,
                -100.0,
                -90.0,
                -100.0,
                -100.0,
            ],
        ]
    }

    # Make the POST request to the biped_hq endpoint
    response = requests.post(url, json=data_to_send)

    # Check if the request was successful
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    main()
