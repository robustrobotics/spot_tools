from openai import OpenAI # Import the OpenAI package
import os # Import the os package
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
import pickle as pkl 
import cv2

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

    # Unpickle the image
    with open("image_dict_1.pkl", "rb") as f:
        image_dict = pkl.load(f)
    
    fl_img = image_dict['frontleft_fisheye_image'][1]
    fr_img = image_dict['frontright_fisheye_image'][1]

    rotated_fl_img = cv2.rotate(fl_img, cv2.ROTATE_90_CLOCKWISE)
    rotated_fr_img = cv2.rotate(fr_img, cv2.ROTATE_90_CLOCKWISE)

    side_by_side = np.hstack((rotated_fr_img, rotated_fl_img))
    cv2.imshow('Side by Side', side_by_side)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Encode the image to base64
    encoded_side_by_side = encode_numpy_image_to_base64(side_by_side)

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Assign the parameters correctly given the image."},
            {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_side_by_side}"
                    }
                },
                {"type": "text", "text": "What's in this image?"}
            ]
        }
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

def test_image_encoding(client):
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        response_format=CalendarEvent,
    )

    event = completion.choices[0].message.parsed
    print(event)

def main(): 
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) # Initialize the OpenAI client with your API key")

    # test_image_encoding(client)
    # # Test the door parameters
    test_door_params(client)

    # Test the image encoding
    # test_image_encoding(client)

    

if __name__ == "__main__":
    main()