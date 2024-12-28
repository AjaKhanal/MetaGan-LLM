import logging
import json

from src import discriminators
import generators


def gan_llama(file_contents, iterations=1):
    logging.basicConfig(level=logging.INFO)
    logging.info(file_contents)

    synthetic = generators.generate_llama(file_contents)
    logging.info(synthetic)


    for i in range(iterations):
        logging.info(f"Iteration {i + 1}:")
        discriminator_response = discriminators.statistical_llama(file_contents, synthetic)
        logging.info(f"Discriminator Response: {discriminator_response}")

        discriminate_json = json.loads(discriminator_response)

        if discriminate_json['Type'] == 'Real':
            break

        print(discriminate_json)
        if "Feedback" in discriminate_json:
            feedback = discriminate_json["Feedback"].strip()
            logging.info("Feedback received by generator")
        else:
            feedback = ""
            logging.info("No feedback provided")

        synthetic = generators.generate_llama(file_contents, feedback=feedback)
        logging.info(synthetic)

    return synthetic
