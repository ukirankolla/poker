pipeline {
    agent {
        docker {
            image 'python:3.12'
            args '-u root'
        }
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    environment {
        PYTHONPATH = '.'
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        REPORT_DIR = 'reports'
    }

    stages {
        stage('Build Pass') {
            steps {
                sh '''
                    pip install -r requirements.txt
                    python -m compileall -q poker agents simulation web tests
                    python -c "import poker.game, poker.evaluator, poker.equity, poker.statistics"
                    python -c "import agents.ollama_agent, agents.learned_agent"
                    python -c "import simulation.benchmark, simulation.tournament, simulation.self_play, simulation.train"
                    python -c "import web.app"
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                sh 'pytest -m "not integration and not regression" --junitxml=${REPORT_DIR}/junit-unit.xml -q'
            }
        }

        stage('Integration Tests') {
            steps {
                sh 'pytest -m integration --junitxml=${REPORT_DIR}/junit-integration.xml -q'
            }
        }

        stage('Regression Tests') {
            steps {
                sh 'pytest -m regression --junitxml=${REPORT_DIR}/junit-regression.xml -q'
            }
        }

        stage('Code Coverage') {
            steps {
                sh 'pytest --cov=poker --cov=agents --cov=simulation --cov=web --cov-report=xml:${REPORT_DIR}/coverage.xml --cov-report=html:${REPORT_DIR}/coverage-html --cov-report=term'
            }
        }

        stage('Publish Reports') {
            steps {
                junit testResults: '${REPORT_DIR}/junit-*.xml', allowEmptyResults: true
                cobertura coberturaReportFile: '${REPORT_DIR}/coverage.xml',
                    coberturaLineCoverage: '85%',
                    coberturaClassCoverage: '85%',
                    coberturaMethodCoverage: '70%',
                    failUnstable: true
                publishHTML(target: [
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'reports/coverage-html',
                    reportFiles: 'index.html',
                    reportName: 'Coverage Report',
                ])
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'reports/**', allowEmptyArchive: true
        }
        success {
            echo 'Pipeline passed: build, unit, integration, regression, and coverage are green.'
        }
        failure {
            echo 'Pipeline failed - see stage logs and report artifacts.'
        }
    }
}
