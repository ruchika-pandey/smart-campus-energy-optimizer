pipeline {
    agent any

    environment {
        PYTHON = 'C:\Users\Ruchika\AppData\Local\Programs\Python\Python314\python.exe'
        PIP = 'C:\Users\Ruchika\AppData\Local\Programs\Python\Python314\Scripts\pip.exe'
    }

    stages {
        stage('Checkout') {
            steps {
                echo '📥 Pulling code from GitHub...'
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                echo '🐍 Installing dependencies...'
                // Use PowerShell to run commands
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
                    # Run pytest and capture results (continue even if tests fail)
                    & ${env.PYTHON} -m pytest tests/ -v --tb=short --junitxml=test-results.xml
                """
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
                // Archive test results
                junit 'test-results.xml'
                // Archive drift report
                archiveArtifacts artifacts: 'drift_report.json', fingerprint: true
                // Archive any model metrics
                archiveArtifacts artifacts: 'model_metrics.json', fingerprint: true
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