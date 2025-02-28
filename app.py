from flask import Flask, request
from service.detect_heart_rate import find_heart_rate
from service.AnalyzeCtScan import analyzeCtScan

app = Flask(__name__)


@app.route('/heart_rate', methods=['POST'])
def detectHeartRate():
    return find_heart_rate(request)


@app.route('/ctscan', methods=['POST'])
def ctScan():
    return analyzeCtScan(request)


if __name__ == '__main__':
    app.run()
