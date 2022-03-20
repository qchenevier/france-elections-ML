set -e # to stop script if something fails

python src/france_elections_ML/pipelines/models/generate_params_and_catalogs.py

kedro docker build --base-image="condaforge/mambaforge"

docker tag $(docker images france-elections-ml:latest -q) 168260601255.dkr.ecr.eu-west-3.amazonaws.com/france-elections-ml

aws ecr get-login-password --region eu-west-3 --no-verify-ssl | docker login --username AWS --password-stdin 168260601255.dkr.ecr.eu-west-3.amazonaws.com

docker push 168260601255.dkr.ecr.eu-west-3.amazonaws.com/france-elections-ml

docker rmi $(docker images -f "dangling=true" -q)
