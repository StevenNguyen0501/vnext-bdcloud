import os
import json

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the serialized JSON string from the environment
LEUKOCYTES = os.getenv("LEUKOCYTES")
NITRITE = os.getenv("NITRITE")
UROBILINOGEN = os.getenv("UROBILINOGEN")
PROTEIN = os.getenv("PROTEIN")
PH = os.getenv("PH")
BLOOD = os.getenv("BLOOD")
GRAVITY = os.getenv("GRAVITY")
KETONE = os.getenv("KETONE")
BILIRUBIN = os.getenv("BILIRUBIN")
GLUCOSE = os.getenv("GLUCOSE")

# Deserialize the JSON string into a dictionary
LEUKOCYTES = json.loads(LEUKOCYTES)
NITRITE = json.loads(NITRITE)
UROBILINOGEN = json.loads(UROBILINOGEN)
PROTEIN = json.loads(PROTEIN)
PH = json.loads(PH)
BLOOD = json.loads(BLOOD)
GRAVITY = json.loads(GRAVITY)
KETONE = json.loads(KETONE)
BILIRUBIN = json.loads(BILIRUBIN)
GLUCOSE = json.loads(GLUCOSE)



print(LEUKOCYTES)
