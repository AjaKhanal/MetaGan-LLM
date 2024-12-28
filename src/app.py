from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

import gan

app = Flask(__name__)
CORS(app)


@app.route('/generate', methods=['GET'])
def generate():
    # Extract the query parameters
    file_content = request.args.get('file_content')
    discriminator = request.args.get('discriminator')
    model = request.args.get('model')  # New model parameter

    # Process the file_content, discriminator, and model here
    result = {
        'message': 'File content processed successfully!',
        'discriminator': discriminator,
        'model': model,  # Include model in the response
        'processed_data': gan.gan_llama(file_content)
    }

    # Print result to server logs
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Processing with model: {model}, discriminator: {discriminator}")
    # print(file_content)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
