pipeline {
    agent any
    triggers {
        githubPush()  // Webhook trigger
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

