pipeline {
    agent any
    environment {
        CI = 'false'
        SSH_KEY_ID = '9f24e7dd-03bf-49c7-a862-dbac775fe926'
        DOCKER_SSH_CONTEXT = 'ec2' // The context name for SSH to EC2
        EC2_USER = 'ubuntu'
        EC2_HOST = '174.129.204.234'
        CONTAINER_NAME = 'genre_detect_api'
        IMAGE_NAME = 'genre_detect_api'
    }
    stages {
        stage('Clone Repo') {
            steps {
                // Clone your repo to get the Dockerfile and application code
                git 'https://github.com/jmrossi98/genre_detect.git'
            }
        }

        stage('Set up Docker Context') {
            steps {
                sshagent(credentials: [SSH_KEY_ID]) {
                    script {
                        // Check if the Docker context already exists
                        def contextExists = sh(script: "docker context ls -q | grep ${DOCKER_SSH_CONTEXT}", returnStatus: true) == 0
                        
                        if (!contextExists) {
                            // Set up Docker context using SSH credentials to access EC2 if it doesn't exist
                            sh 'docker context create ${DOCKER_SSH_CONTEXT} --docker "host=ssh://${EC2_USER}@${EC2_HOST}"'
                        } else {
                            sh 'docker context update ${DOCKER_SSH_CONTEXT} --docker "host=ssh://${EC2_USER}@${EC2_HOST}"'
                        }
                    }
                }
            }
        }

        stage('Stop Container') {
            steps {
                sshagent(credentials: [SSH_KEY_ID]) {
                    script {
                        sh '''
                        if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
                        echo "Stopping container $CONTAINER_NAME..."
                        docker stop $CONTAINER_NAME
                        else
                        echo "No running container named $CONTAINER_NAME found."
                        fi
                        '''
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sshagent(credentials: [SSH_KEY_ID]) {
                    script {
                        // Use the created context to build the Docker image
                        sh 'docker context use ${DOCKER_SSH_CONTEXT}'
                        sh 'docker build -t ${IMAGE_NAME} .'
                    }
                }
            }
        }
        
        stage('Deploy Docker Container') {
            steps {
                sshagent(credentials: [SSH_KEY_ID]) {
                    script {
                        // Use Docker to deploy the container
                        sh 'docker run -p 5000:5000 -d --name ${CONTAINER_NAME} --rm ${CONTAINER_NAME}'
                    }
                }
            }
        }
    }
}