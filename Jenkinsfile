pipeline {
    agent any

    environment {
        PYTHON = 'C:\\Users\\Ruchika\\AppData\\Local\\Programs\\Python\\Python314\\python.exe'
        PIP = 'C:\\Users\\Ruchika\\AppData\\Local\\Programs\\Python\\Python314\\Scripts\\pip.exe'
    }

    stages {
        stage('Clean Workspace') {
            steps {
                cleanWs()  // deletes the workspace before checkout
            }
        }

        stage('Checkout') {
            steps {
                echo '📥 Pulling code from GitHub...'
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                echo '🐍 Installing dependencies...'
                powershell """
                    & ${env.PYTHON} --version
                    & ${env.PIP} install --upgrade pip
                    & ${env.PIP} install -r requirements.txt
                    & ${env.PIP} install pytest mlflow pandas numpy scikit-learn
                """
            }
        }

        stage('Run Tests') {
            steps {
                echo '🧪 Running unit tests...'
                powershell """
                    & ${env.PYTHON} -m pytest tests/ -v --tb=short --junitxml=test-results.xml
                """
            }
        }

        stage('Debug - List Files') {
            steps {
                echo '📂 Listing workspace contents...'
                powershell 'Get-ChildItem -Recurse -Path .'
            }
        }

        stage('Check Model Drift') {
            steps {
                echo '📊 Checking for drift...'
                powershell """
                    & ${env.PYTHON} mlops/drift_detection.py
                """
            }
        }

        stage('Archive Artifacts') {
            steps {
                echo '💾 Saving reports...'
                junit 'test-results.xml'
                archiveArtifacts artifacts: 'drift_report.json', fingerprint: true
                // archiveArtifacts artifacts: 'model_metrics.json', fingerprint: true (commented out)
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully!'
            // Optional: send email notifications
        }
        failure {
            echo '❌ Pipeline failed. Check the console log.'
        }
    }
}