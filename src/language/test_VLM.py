from openai import OpenAI # Import the OpenAI package
import os # Import the os package
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np

def encode_numpy_image_to_base64(image_np: np.ndarray, format: str = "PNG") -> str:

    pil_image = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)

    b64_bytes = base64.b64encode(buffer.read())
    b64_string = b64_bytes.decode("utf-8")
    return b64_string

def test_door_params(client):
    class OpenDoorParams(BaseModel):
        hinge_side: int          
        swing_direction: int 
        handle_type: int 

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the information about the door skill."},
            {"role": "user", "content": "The door hinge is on the left (1), the swing direction is unknown (0), and the door handle is unknown (0)."},
        ],
        response_format=OpenDoorParams,
    )

    parameters = completion.choices[0].message.parsed
    print(parameters)

def test_image_encoding(client):
    # Create a dummy image that is all a single random color 
    image_np = np.ones((100, 100, 3), dtype=np.uint8) * np.random.randint(0, 255, size=(1, 1, 3), dtype=np.uint8)
    
    # Visualize the image 
    image = Image.fromarray(image_np)
    image.show()

    # Encode the image to base64
    encoded_image = encode_numpy_image_to_base64(image_np)
    
    # Use the OpenAI API to process the image
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
    )

    print(completion.choices[0].message.content)

def main(): 
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) # Initialize the OpenAI client with your API key")

    # # Test the door parameters
    # test_door_params(client)

    # Test the image encoding
    test_image_encoding(client)

    

if __name__ == "__main__":
    main()