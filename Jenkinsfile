pipeline {
    agent any
    triggers {
        githubPush()  // Listens for GitHub webhook events
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git branch'  // Debug: Show branch name
            }
        }
    }
}
