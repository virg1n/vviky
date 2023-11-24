import baseten
import base64

# baseten.login("rqaniThl.Zxo62MAICRl7IiyY3W73u4UcdhL6x99F")

def generateImage(name, dir, prompt="dog", w=1024, h=1024, negative_prompt="letters"):

    model = baseten.deployed_model_version_id('wd1mrew')

    request = {
        "prompt": prompt,
        "use_refiner": False,
        "steps":20,
        "negative_prompt":"letters, many images",
        "width":w, 
        "height":h,
    }

    response = model.predict(request)

    img=base64.b64decode(response["data"])

    img_file = open(f'{dir}/{name}.jpeg', 'wb')
    img_file.write(img)
    img_file.close()


# generateImage("frog", dir="D")