import os

from static import launch_commands
from voice import recognize_speech


# Function to launch script
def launch_script():
    os.system('python player.py')


# Main loop
while True:
    command = recognize_speech()
    if command in launch_commands:
        launch_script()
