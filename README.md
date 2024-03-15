# n-vibe-cluster-api

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
8. Run the docker : `docker run --mount type=bind,src=~/weights,dst=/app/models/weights -d -p 3000:3000 n-vibe-cluster-api`

> [!NOTE]
> `--mount type=bind,src=dir,dst=dir` binds a folder from the SSH machine to the inside of the docker \
> `-d -p 3000:3000` opens the port 3000 of the docker to listen to the port 3000 of the SSH machine
