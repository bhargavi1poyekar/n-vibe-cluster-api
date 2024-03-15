import requests
from app.utils.utils import time_and_log


@time_and_log
def main():
    # The URL for the API endpoint (assuming the Flask app is running locally on port 5000)
    url = "http://34.247.213.126:3000/biped/cluster"
    # url = "http://127.0.0.1:3000/biped/cluster"

    # Example data to send in the request
    # Replace this with the actual RSSI signals data you want to send
    # Biped requires 10 RSSI values per sample
    data_to_send = {
        "tab": [
            [-66, -72, -68, -70, -76, -89, -100, -79, -70, -76],
            [-66, -73, -68, -68, -76, -89, -100, -81, -70, -77],
            [-66, -75, -71, -66, -75, -90, -100, -81, -67, -77],
        ]
    }

    def change_order(data_to_send: dict):
        new_data = {"tab": []}
        for tab in data_to_send["tab"]:
            new_tab = [
                tab[9],
                tab[8],
                tab[7],
                tab[4],
                tab[3],
                tab[1],
                tab[0],
                tab[2],
                tab[6],
                tab[5],
            ]
            new_data["tab"].append(new_tab)
        return new_data

    # Make the POST request to the biped_hq endpoint
    response = requests.post(url, json=change_order(data_to_send))

    # Check if the request was successful
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    main()
