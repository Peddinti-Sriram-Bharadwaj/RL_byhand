pipeline {
    agent any
    triggers { pollSCM('H/5 * * * *') } // Poll every 5 minutes
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git branch'  // Debug: Show branch name
            }
        }
    }
}

