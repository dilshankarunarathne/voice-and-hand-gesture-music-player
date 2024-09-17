import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()


# Function to recognize speech
def recognize_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(text)
            return text.lower()
        except:
            return ""


def parse_voice_command() -> str:
    com = recognize_speech()
    if com == "":
        return ""
    else:
        return com
