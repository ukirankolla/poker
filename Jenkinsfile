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
        IMAGE = 'ghcr.io/ukirankolla/poker'
    }

    stages {
        /* ---- CI: build verification ---- */
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

        /* ---- CI: unit tests ---- */
        stage('Unit Tests') {
            steps {
                sh 'pytest -m "not integration and not regression" --junitxml=${REPORT_DIR}/junit-unit.xml -q'
            }
        }

        /* ---- CI: integration tests ---- */
        stage('Integration Tests') {
            steps {
                sh 'pytest -m integration --junitxml=${REPORT_DIR}/junit-integration.xml -q'
            }
        }

        /* ---- CI: regression tests ---- */
        stage('Regression Tests') {
            steps {
                sh 'pytest -m regression --junitxml=${REPORT_DIR}/junit-regression.xml -q'
            }
        }

        /* ---- CI: code coverage ---- */
        stage('Code Coverage') {
            steps {
                sh 'pytest --cov=poker --cov=agents --cov=simulation --cov=web --cov-report=xml:${REPORT_DIR}/coverage.xml --cov-report=html:${REPORT_DIR}/coverage-html --cov-report=term'
            }
        }

        /* ---- CI: publish reports ---- */
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

        /* ---- CD: build Docker image ---- */
        stage('Docker Build') {
            agent {
                docker {
                    image 'docker:24'
                    args '-u root -v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            steps {
                sh """
                    docker build \
                        -t ${IMAGE}:${env.BUILD_NUMBER} \
                        -t ${IMAGE}:latest \
                        .
                """
            }
        }

        /* ---- CD: push image to GitHub Container Registry ---- */
        stage('Docker Push') {
            when {
                branch 'main'
            }
            agent {
                docker {
                    image 'docker:24'
                    args '-u root -v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'ghcr-token',
                    usernameVariable: 'GHCR_USER',
                    passwordVariable: 'GHCR_PASS'
                )]) {
                    sh """
                        echo \$GHCR_PASS | docker login ghcr.io -u \$GHCR_USER --password-stdin
                        docker push ${IMAGE}:${env.BUILD_NUMBER}
                        docker push ${IMAGE}:latest
                    """
                }
            }
        }

        /* ---- CD: deploy to target host via SSH ---- */
        stage('Deploy') {
            when {
                branch 'main'
            }
            agent {
                docker {
                    image 'docker:24'
                    args '-u root -v /var/run/docker.sock:/var/run/docker.sock'
                }
            }
            environment {
                DEPLOY_HOST = credentials('deploy-host')
            }
            steps {
                sshagent([env.DEPLOY_HOST]) {
                    sh """
                        ssh -o StrictHostKeyChecking=no ${env.DEPLOY_USER}@${env.DEPLOY_HOST} \\
                            "cd /opt/poker && docker compose pull && docker compose up -d"
                    """
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'reports/**', allowEmptyArchive: true
        }
        success {
            echo 'Pipeline passed: build, unit, integration, regression, coverage, and container image are green.'
        }
        failure {
            echo 'Pipeline failed - see stage logs and report artifacts.'
        }
    }
}
