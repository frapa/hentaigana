import argparse
from pathlib import Path
from typing import List, Union
import os

import numpy as np
import tensorflow as tf

INPUT_SIZE = (228, 128)
BATCH_SIZE = 8
DATA_PATH = Path(__file__).parent / "data"
OUTPUT_PATH = Path(__file__).parent / "models" / "model"
OUTPUT_MAP = {
    0: "unclassified",
    1: "tsu TADA",
    2: "ke NOZOMU",
    3: "no DEKIRU",
    4: "ha IPPAN",
    5: "ke KUUKI",
    6: "ni TANNEN",
    7: "tsu TSUNAMI",
}


def main():
    args = _parse_args()
    args.func(args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=_train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.set_defaults(func=_predict)
    predict_parser.add_argument("model_path", type=Path)
    predict_parser.add_argument("image_path", type=Path)

    return parser.parse_args()


def _load_image(path: Union[Path, tf.Tensor]) -> tf.Tensor:
    if isinstance(path, Path):
        path = str(path)

    return tf.io.decode_image(
        tf.io.read_file(path),
        # https://stackoverflow.com/questions/44942729/tensorflowvalueerror-images-contains-no-shape
        expand_animations=False,
    )


def _extract_class(path: tf.Tensor) -> tf.Tensor:
    folder = tf.strings.split(path, os.path.sep)[-2]
    return tf.strings.to_number(folder)


def _resize_and_crop(img: tf.Tensor) -> tf.Tensor:
    # h, w = tf.shape(img)
    # size = tf.cond(h > w, (128 / w * h, 128), (128, 128 / h * w))
    size = INPUT_SIZE
    resized = tf.image.resize(img, size, preserve_aspect_ratio=True)
    return resized
    # crop = tf.image.crop_to_bounding_box(resized, 55, 0, 128, 128)
    # return crop


def _prepare_input(path: Union[Path, tf.Tensor]) -> tf.Tensor:
    image = _load_image(path)
    grayscale = tf.image.rgb_to_grayscale(image)
    crop = _resize_and_crop(grayscale)
    normalized = 1 - tf.cast(crop, tf.float32) / 255
    return normalized


def _augment(data):
    img, label = data
    return tf.image.random, label


def _make_dataset() -> tf.data.Dataset:
    glob = str(DATA_PATH / "**/*.jpeg")

    return (
        tf.data.Dataset.list_files(glob)
        .repeat(4)
        .shuffle(1000)
        .map(
            lambda path: (
                # Image
                _prepare_input(path),
                # Target class, folder name (0 for unclassified)
                _extract_class(path),
            ),
            num_parallel_calls=4,
        )
        .batch(BATCH_SIZE)
    )


def _make_callbacks() -> List[tf.keras.callbacks.Callback]:
    OUTPUT_PATH.mkdir(parents=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=OUTPUT_PATH / "epoch_{epoch:03d}-loss_{loss:.5f}",
        monitor="loss",
        verbose=True,
        save_best_only=True,
    )

    return [checkpoint]


def _make_model() -> tf.keras.models.Model:
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer((*INPUT_SIZE, 1)),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
            tf.keras.layers.Conv2D(
                kernel_size=3,
                strides=2,
                filters=16,
                activation="relu",
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.Conv2D(
                kernel_size=3,
                strides=2,
                filters=32,
                activation="relu",
                kernel_initializer="he_uniform",
            ),
            # tf.keras.layers.Conv2D(
            #     kernel_size=3,
            #     filters=32,
            #     activation="relu",
            #     kernel_initializer="he_uniform",
            # ),
            tf.keras.layers.Conv2D(
                kernel_size=3,
                strides=2,
                filters=64,
                activation="relu",
                kernel_initializer="he_uniform",
            ),
            # tf.keras.layers.Conv2D(
            #     kernel_size=3,
            #     filters=64,
            #     activation="relu",
            #     kernel_initializer="he_uniform",
            # ),
            tf.keras.layers.Conv2D(
                kernel_size=3,
                strides=2,
                filters=128,
                activation="relu",
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.Conv2D(
                kernel_size=3,
                strides=2,
                filters=len(OUTPUT_MAP),
                activation="relu",
                kernel_initializer="he_uniform",
            ),
            tf.keras.layers.GlobalMaxPool2D(),
            tf.keras.layers.Softmax(),
        ]
    )


def _train(_args: argparse.Namespace):
    dataset = _make_dataset()

    model = _make_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(dataset, epochs=100, callbacks=_make_callbacks())


def _predict(args: argparse.Namespace):
    result = predict(args.model_path, args.image_path)
    print(result)


def predict(model_path: Path, image_path: Path) -> str:
    image = _prepare_input(image_path)

    model = tf.keras.models.load_model(model_path)
    result = model.predict(image[None])[0]

    cls = np.argmax(result)
    return OUTPUT_MAP[cls]


if __name__ == "__main__":
    main()
