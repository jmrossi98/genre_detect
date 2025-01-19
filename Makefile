deploy-image:
	docker container stop "my-flask-api"
	docker image rm "my_flask_api"
	docker build -f build/Dockerfile -t my_flask_api .
	docker run -p 5000:5000 -d --name my-flask-api --rm my_flask_api:latest

deploy-cluster:
	kubectl apply -f build/flask-deployment.yaml
	kubectl apply -f build/flask-service.yaml

