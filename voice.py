import tkinter as tk
from tkinter import ttk
import threading
import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

# Function to recognize speech
def recognize_speech(max_retries=3):
    with sr.Microphone() as source:
        retries = 0
        while retries < max_retries:
            print("Listening...")
            try:
                audio = r.listen(source, timeout=3)
                text = r.recognize_google(audio)
                print(text)
                return text.lower()
            except sr.WaitTimeoutError:
                print("Listening timeout, retrying...")
                retries += 1
            except sr.UnknownValueError:
                print("Could not understand audio")
                retries += 1
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return ""
        print("Max retries reached, stopping listening.")
        return ""

def parse_voice_command() -> str:
    com = recognize_speech()
    if com == "":
        return ""
    else:
        return com