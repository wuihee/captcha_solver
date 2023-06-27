import base64
from pathlib import Path
import pathlib

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CaptchaBreaker:
    def __init__(self):
        self.model = keras.models.load_model("C:/Users/wuihee/Desktop/Programming/Projects/CAPTCHA Solver/model")

        characters = ['2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
        self.num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    def encode_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.transpose(image, perm=[1, 0, 2])
        return image

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :1]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def load_captcha(self, image_data):
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        for i in range(4):
            col = image[:, i * 25:(i + 1) * 25]
            cv2.imwrite(f"./test_captcha/letter{i}.png", col)

        data_dir = pathlib.Path('./test_captcha')
        data_dir = [str(path) for path in data_dir.iterdir()]

        dataset = tf.data.Dataset.from_tensor_slices(data_dir)
        dataset = (
            dataset.map(self.encode_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(16)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return dataset

    def solve(self, image_data):
        dataset = self.load_captcha(image_data)
        predictions = self.model.predict(dataset)
        answer = ''.join(self.decode_batch_predictions(predictions))
        return answer


image_data = "iVBORw0KGgoAAAANSUhEUgAAAGQAAAAoCAIAAACHGsgUAAACJUlEQVR42u3asUoEQQwG4KsFG6sDQQTfwMrisBRsLHwB2xMbxdfwHeysfAhLO1/h3uQcCIzjJJv9ZzazNwuBFMexcOx3M0kmu6v9/tEDjJUTOJZjOZZjOZZjOYFjOZZj6bF9e20Xy8C6O9+l0eGfdPvw7lgNsI6Or3jMILW9WXcrJWCJTIZkmRRfVsGLk23WGyQWhnV99oVIZZ+FvP6frEcshWO6V6aDJKxI1unKUgLfidyL04xifX9epGSGFiHuP0546FJzY+FSqRetGqvcLzKlXnNjZV7cpQLr9PIlxBQynckSq6IakpfogndYwYuY4jYUy2UFlniBAVZ16xC8RBQEi4wClpi/i8ienn+KogDLqsMiDrE46ljEFJsG8qq+c7p5ZUEhdbAMq7r5RCpjahSZMiwlRpdYtvuG6qAZVilZypF5cSxulLWjiJdChqT20YQF5awKL87Bi2MIcSkNHXR0Lz2RgVi6VJNqOLTLoldkQn43xUJHYIxMr4Mplp7UjLH4OTmN4EVM4IgmWzi4V0amdwzIBQfAIq/RwYMJVko2Kxaes5pi1XkR2QGwTCZ/5FWUsCZipUtsEpauYD4sVTpVBKvaiyDEcglK/WEhYTtQHvVSGoWsjSg9G/JEZollPn1HxqogVvVZGsz9Qs5qzWSLlXpNHD8oiay7h6xg8lLmqAbPceEBxuGfSPfgBZIt+F0HWyyEbNkvhrTwUsj8LRqNLPvmFzSTJ2tW3jLKAAAAAElFTkSuQmCC"
captcha_breaker = CaptchaBreaker()
answer = captcha_breaker.solve(image_data)
print(answer)
