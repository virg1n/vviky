import baseten

# You can retrieve your deployed model version ID from the UI
model = baseten.deployed_model_version_id('wd1mrew')

request = {
    "prompt": "Frogs anatomy",
    "use_refiner": False,
    "steps":20
}

response = model.predict(request)

import base64

img=base64.b64decode(response["data"])

img_file = open('image.jpeg', 'wb')
img_file.write(img)
img_file.close()