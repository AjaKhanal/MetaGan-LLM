from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI
from ollama import ChatResponse, chat
import re


# TODO Make better
def self_rubric(input_string):
    """
    Creates a regex pattern from a given string by escaping special characters.

    :param input_string: The string to convert into a regex pattern.
    :return: A regex pattern object.
    """
    # Escape special characters in the string
    escaped_string = re.escape(input_string)
    # Compile the escaped string into a regex pattern
    regex_pattern = re.compile(escaped_string)
    return regex_pattern


def generate_gpt(file_contents, _model="gpt-4o-mini", feedback=""):
    rubric = self_rubric(file_contents)
    key = load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=os.environ.get('OPENAI_API_KEY'),
    )

    completion = client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": "You are a data analyst"},
            {
                "role": "user",
                "content": (
                    f"Identify the metadata for the following dataset: \n\n{file_contents}\n\n"
                    "Classify the patterns you observe and generate 100 additional synthetic rows to augment the dataset. No python code"
                    "Ensure the new rows follow the same structure of the filetype, maintain realistic distributions, and include plausible values for each column."
                    "Just give the generated data in code blocks. Nothing else"
                    "Include CSV headers if provided"
                    f"Use this sample regex as a rubric \n\n{rubric}\n\n"
                    f"Here is some feedback you can use to make this augmented dataset better: {feedback}"
                )
            }
        ]
    )

    ans = completion.choices[0].message.content
    return ans[3:-3]


def generate_llama(file_contents, _model="llama3.1", feedback=""):
    rubric = self_rubric(file_contents)
    response: ChatResponse = chat(
        model=_model,
        messages=[
            {"role": "system", "content": "You are a data analyst"},
            {
                "role": "user",
                "content": (
                    f"Identify the metadata for the following dataset: \n\n{file_contents}\n\n"
                    "Classify the patterns you observe and generate 200 additional synthetic rows to augment the dataset. No python code"
                    "Ensure the new rows follow the same structure of the filetype, maintain realistic distributions, and include plausible values for each column."
                    "Only provide generated data in a single code block (start/end with ```). Nothing else. Do not write anything else that's not generated data, no explanations!"
                    "Include CSV headers if provided"
                    f"Use this sample regex as a rubric \n\n{rubric}\n\n"
                    f"Here is some feedback you can use to make this augmented dataset better: {feedback}"
                )
            }
        ]
    )

    ans = response.message.content
    return ans[3:-3]
