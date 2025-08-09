import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from PyPDF2 import PdfReader
import gradio as gr
from pydantic import BaseModel
import requests


load_dotenv(override=True)

sendgrid_client = os.getenv("SENDGRID_API_KEY")
sendgrid_user = os.getenv("PUSHOVER_USER")
sendgrid_token = os.getenv("PUSHOVER_TOKEN")

def Pushover(message):
    return requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": sendgrid_token,
            "user": sendgrid_user,
            "message": message,
        },
    )

def record_user_details(email, name="Not set", note="Not set"):
    Pushover(f"User {name} with email {email} has been recorded with note {note}")
    print(f"User {name} with email {email} has been recorded with note {note}")
    return {"status": "success"}

def record_unknown_question(question):
    Pushover(f"Unknown question: {question}")
    print(f"Unknown question: {question}")
    return {"status": "success"}

record_unknown_question_tool_structure = {
    "name": "record_unknown_question",
    "description": "Use this tool to record an unknown question asked by the user",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to record"
            }
        }, 
        "required": ["question"]
    }
}

record_user_details_tool_structure = {
    "name": "record_user_details",
    "description": "Use this tool to record the details of the user that is interested in getting in touch",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email of the user"
            },
            "name": {
                "type": "string",
                "description": "The name of the user"
            },
            "note": {
                "type": "string",
                "description": "The note of the user"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_tool_structure}, {"type": "function", "function": record_unknown_question_tool_structure}]

class Chat(BaseModel):
    isAccepted: bool
    reason: str


class Me:
    def __init__(self):
        reader = PdfReader("cv.pdf")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.gemini_client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai")
        
        self.name = "Damilare Bakare"
        self.summary = ""
        self.cv = ""

        for page in reader.pages:
            self.cv += page.extract_text()
        with open("summary.txt", "r") as file:
            self.summary = file.read()

    def get_system_prompt(self) -> str:
       return f"""
            You are a helpful assistant that can answer questions about {self.name}.

            You are given a summary of {self.name} and {self.name}'s CV. you need to answer questions about {self.name}.

            The summary is:
            {self.summary}

            The CV is:
            {self.cv}

            As {self.name}'s personal assistant, you must be polite, professional and engaging.

            You need to answer the questions based on the summary and the CV.

            You have the ability to use record_user_details tool to record the details of the user that is interested in getting in touch. and also record_unknown_question tool to record the unknown question asked by the user.

            Make sure to nudge the user to the direction of dropping their details to get in touch with {self.name}.

            """

    def evaluate(self, history, question, response)->Chat:

        evaluator_prompt = f"""
        You are a response evaluator. Your task is to evaluate the response of another assistant to a chat about {self.name}. 
        You are to ensure that the response is relevant to the chat, not of the context, professional and engaging.
        Given the {self.name}'s summary to be {self.summary} and the {self.name}'s CV to be {self.cv}, chat history to be {history} and the question to be {question}, you need to evaluate the assistant's repoonse {response} to be sure that it is relevant to the chat, not of the context, professional and engaging.
        """

        evaluator_message = [{
            "role": "user",
            "content": evaluator_prompt
        }]

        evaluator_response = self.gemini_client.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=evaluator_message,
            response_format=Chat
        )

        return evaluator_response.choices[0].message.parsed

    def reRun(self, feedback, history, question)->str:
        rerun_prompt = f"""
        You are a response rerunner. Your task is to rerun the response of another assistant to a chat about {self.name}. 
        You are to ensure that the response is relevant to the chat, not of the context, professional and engaging.
        Given the {self.name}'s summary to be {self.summary} and the {self.name}'s CV to be {self.cv}, chat history to be {history} and the question to be {question}, 
        You need to generate another response based on the feedback {feedback} to be sure that it is relevant to the chat, not of the context, professional and engaging.
        """

        rerun_message = [{
            "role": "system",
            "content": rerun_prompt
        }] + history + [
            {
                "role": "user",
                "content": question
            }
        ]

        rerun_response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=rerun_message
        )

        return rerun_response.choices[0].message.content
    
    def handle_tool_call(self, tool_calls):
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"Tool Called: {tool_name} with args: {tool_args}")
            tool = globals()[tool_name]
            tool_result = tool(**tool_args) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(tool_result),
                "tool_call_id": tool_call.id
            })

        # Return the list of tool result messages for inclusion in the next model call
        return results
    
    def chat(self, message, history):
        # Build message list once and keep appending
        messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ] + history + [
            {"role": "user", "content": message}
        ]

        max_tool_loops = 5
        loops = 0

        while True:
            print(messages)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
            )

            choice = response.choices[0]
            finish_reason = choice.finish_reason
            print(finish_reason)

            assistant_message = choice.message

            # If the model requests tools, append the assistant's tool_calls and the tool results, then loop
            if finish_reason == "tool_calls" and assistant_message.tool_calls:
                loops += 1
                if loops > max_tool_loops:
                    return "Sorry, I ran into a tool loop. Please try again."

                # Add assistant message with tool_calls
                messages.append(assistant_message)

                # Execute tools and add their outputs as tool messages
                tool_messages = self.handle_tool_call(assistant_message.tool_calls)
                messages.extend(tool_messages)
                continue

            # Otherwise, we have a normal assistant response
            break

        unevaluated_response = assistant_message.content or ""

        evaluator_result = self.evaluate(history, message, unevaluated_response)
        print(evaluator_result)
        if evaluator_result.isAccepted:
            print("Accepted")
            return unevaluated_response
        else:
            print("Re-running")
            return self.reRun(evaluator_result.reason, history, message)


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, title="Damilare Bakare's Assistant", type="messages").launch()