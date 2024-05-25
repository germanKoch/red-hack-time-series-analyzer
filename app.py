from flask import Flask, request, jsonify
from datetime import datetime
from backend.service.anomaly_analizer_service import AnomalyAnalizerService
from backend.service.types import TimeSeriesType
from flask_cors import CORS


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, origins="*")

# @app.before_request
# def before_request():
#     headers = {'Access-Control-Allow-Origin': '*',
#                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
#                'Access-Control-Allow-Headers': 'Content-Type'}
#     if request.method.lower() == 'options':
#         return jsonify(headers), 200

service = AnomalyAnalizerService()

def parse_datetime(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    
@app.route('/get-series', methods=['GET'])
def get_series():
    option = request.args.get('time-series')
    series_type = TimeSeriesType[option]
    return jsonify(service.get_series(series_type))


@app.route('/get-anomilies', methods=['GET'])
def get_anomalies():
    start = request.args.get('start')
    end = request.args.get('end')
    option = request.args.get('time-series')

    # Parse and validate the datetimes
    start_datetime = parse_datetime(start)
    end_datetime = parse_datetime(end)
    series_type = TimeSeriesType[option]

    result = service.get_anomaly(series_type, start_datetime, end_datetime)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
