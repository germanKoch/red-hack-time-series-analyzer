from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/endpoint', methods=['GET'])
def endpoint():
    return jsonify({"message": "Hello, World!"})

if __name__ == '__main__':
    app.run(debug=True)
