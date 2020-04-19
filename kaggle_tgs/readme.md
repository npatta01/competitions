
gcloud compute --project "np-training" ssh --zone "us-east1-c" "ubuntu@dl" -- -L 8888:localhost:8080


git config --global credential.helper 'cache --timeout 2880000'
