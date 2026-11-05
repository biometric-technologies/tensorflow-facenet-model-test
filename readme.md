![](.github/readme/banner.svg)

# FaceNet TensorFlow Lite model test

A small Python script for checking a FaceNet face recognition model in TensorFlow Lite format on your own photos. FaceNet turns a face into an embedding, a fixed list of numbers, and two photos of the same person give embeddings that lie close together. Point the script at folders of face photos, one folder per person, and it prints the distance for every pair, so you can see how well the model separates people before you ship it in an app. The repository includes a sample `model.tflite` and four sample folders under `images/`.

## What it does

- Loads a `.tflite` model with the TensorFlow Lite interpreter and prints its input and output tensor shapes.
- Finds the face in each photo with OpenCV's Haar cascade, crops it and resizes it to 160 x 160.
- Normalizes pixels as `(x - 127.5) / 128`, runs the model and takes the output as the face embedding.
- Compares every pair of photos inside a folder (same person) and each photo against a random photo from another folder (different person), using the Euclidean distance between embeddings.
- Prints every score and a summary at the distance threshold 1.0.

## How to run

Python 3.7 with the versions pinned in `requirements.txt` (TensorFlow 2.6.0, OpenCV 4.5.3, NumPy 1.21.2, Pillow 8.3.1).

```bash
pip3 install -r requirements.txt
python3 main.py
```

Settings live in the `args` dictionary at the top of `main.py`: the model file (`model.tflite`), the threshold (`1.0`), the images folder (`./images`) and whether to print every score. Arrange your photos like this, with at least two photos per person and `.jpg`, `.jpeg` or `.png` files:

```text
images/
├── 1/   photo-a.jpg, photo-b.jpg ...
├── 2/   ...
└── 3/   ...
```

The per-pair scores in the log are the useful output: same-person pairs should fall below the threshold, different-person pairs above it. Two things to know about the summary lines: the script treats a same-person distance above the threshold as a false reject, and it also adds different-person distances below the threshold to the same false-reject counter, so the printed false accept rate stays at 0; and a photo in which the Haar cascade finds no face is skipped with a warning.

## Related repositories

- [biometric-sdk](https://github.com/biometric-technologies/biometric-sdk): the SDK that runs a FaceNet-type TensorFlow Lite model in production, configured through `FaceNetModelConfiguration(path, inputWidth, inputHeight, outputLength)`.
- [facenet-recognition-model](https://github.com/biometric-technologies/facenet-recognition-model): reserved for the FaceNet model itself.
- [biometric-models-onnx-converters](https://github.com/biometric-technologies/biometric-models-onnx-converters): converts FaceNet and the liveness model to ONNX.
- [liveness-detection-model](https://github.com/biometric-technologies/liveness-detection-model): the liveness check that runs before face matching.

## License

License: not yet specified. The repository has no license file; contact Iriscan before reusing the code.

## Contact

info@iriscan.com, https://www.iriscan.com
