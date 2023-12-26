
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)

@app.route('/my-backend-endpoint', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '')

    # Call the Python script with the query string as an argument
    result = subprocess.run(
        ['python3', './main.py', query],
        stdout=subprocess.PIPE,
        text=True
    ).stdout.splitlines()

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)

