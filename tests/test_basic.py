from hate_speech_detection import model_gpt2, train_bert
from hate_speech_detection.models import models


def test_example():
    assert 1 == 1


def test_models():
    print("test_models**")
    print(models)
    assert 1 == 1


def test_models_working():
    print("test_models_working**")
    for k in models.keys():
        print(models[k].predict(["Battery draining quickly!",
              "This watch is excellent"]))
    assert 1 == 1
