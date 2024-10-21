from resume_matcher import talk_to_openai


def test_talk_to_openai_text_only():
    response = talk_to_openai("Hello, OpenAI", max_tokens=100)
    assert isinstance(response, str)
    assert len(response) > 0

def test_talk_to_openai_with_image():
    with open("wahle.png", "rb") as img_file:
        image_data = [img_file.read()]
    response = talk_to_openai("Hello, OpenAI with image", image_data=image_data)
    assert isinstance(response, str)
    assert len(response) > 0
