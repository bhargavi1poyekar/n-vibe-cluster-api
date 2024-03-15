# n-vibe-cluster-api

This repository is an API for ML models intended to indoor localisation for the N-Vibe application.

## Used architecture

The API is made using Flask blueprints to redirect and standardise usage based on the location's name.
Currently, the N-Vibe office is located at Mairie XX's headquarters. Thus, the URL endpoint is :

```bash
http://<server_url>:3000/mairie-xx/<request>
```

The available requests are showed as `requests`, whereas future work is mentionned as _request_ :

- `cluster` : predicts the cluster based on RSSI signals
- _map_ : returns elements of the map
- _logs_ : returns logs of all the different endpoint usage

## Adding a new model

If you want to add a new model, you will have to develop a new endpoint for the API in the correct adress. For example, if you want to add a stairs detection model in the `mairie_xx`'s location, add a new endpoint to handle the stairs model in the `api/app/routes/mairie_xx.py` file.

Add your model to `api/app/models/weights/mairie_xx`, and name it with an clear and straightforward name.

Finally, you need to add the reference to your model in the `api/app/config/mairie_xx.yaml` file. This file holds all variables for a single map. Feel free to create a new key, or modify the structure of the model. You will need to get this key, either by adding a new function to get a specific key of a configuration file, or by modifying the `get_model` function in the `utils/model.py` file.

Please make sure that _all_ API endpoints work after your modifications. The test files are made for this purpose. You can either :

- launch the server locally test all endpoints
- use the server's URL to query the currently running Docker

In progress : _We will soon be using the `pytest` framework to validate any server modifications. Please understand that you have to understand and develop test cases for *all* API endpoints. This ensures that when the docker is uplaoded and ran, each function is assured to work as intended._

## Docker installation

1. Log into the SSH machine
2. Stop the previously running docker
   1. See which docker is running : `docker ps -a`
   2. Retain the name of the docker corresponding to the image `n-vibe-cluster-api:latest`
3. Stop the docker container : `docker stop your_container_name`
4. Remove the docker container : `docker rm your_container_name`
5. Navigate to the `n-vibe-cluster-api` folder
6. `git pull` if anything changed
7. Build the docker from the pulled repository : `docker build -t n-vibe-cluster-api .`
8. Run the docker : `docker run --mount type=bind,src=/home/ubuntu/n-vibe-cluster-api/api/app/models/weights,dst=/api/app/models/weights -d -p 3000:3000 n-vibe-cluster-api`

> [!NOTE] > `--mount type=bind,src=source_dir_path,dst=destination_dir_path` binds a folder from the SSH machine to the inside of the docker \
> `-d -p 3000:3000` opens the port 3000 of the docker to listen to the port 3000 of the SSH machine

## Inspect the docker

To connect to the docker deamon, run :

```bash
docker exec -it <container_name> bash
```
