import tensorflow as tf
import cv2
import numpy as np
import os
import itertools
import random

args = {
    'model': 'model.tflite',
    'threshold': 1.0,
    'images_folder': './images',
    'print_log': True
}

def load_tflite_model(tflite_model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


def run_facenet_model(image, interpreter, input_details, output_details):
    img = cv2.resize(image, (160, 160))
    img = img.astype(np.float32)
    img = (img - 127.5) / 128  # normalize
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def extract_face(img):
    faces = detector.detectMultiScale(img, 1.1, 4)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (160, 160))
        return face
    else:
        return None


def load_rnd_image_from_folder(folder):
    images = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_path = os.path.join(subdir, file)
                img = cv2.imread(image_path)
                face = extract_face(img)
                if face is not None:
                    to_append = (image_path, face)
                    images.append(to_append)
    idx = random.randint(0, len(images) - 1)
    return images[idx]


def load_images_pairs_from_folder(base_folder):
    images = []
    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_path = os.path.join(subdir, file)
                img = cv2.imread(image_path)
                face = extract_face(img)
                if face is not None:
                    to_append = (image_path, face)
                    images.append(to_append)
                else:
                    print(f"WARN: Face was not found on image {image_path}")
    return itertools.combinations(images, 2)


if __name__ == '__main__':

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    tflite_model_path = args['model']
    interpreter, input_details, output_details = load_tflite_model(tflite_model_path)
    for i, input_detail in enumerate(input_details):
        print(f"Input tensor {i}: shape = {input_detail['shape']}")
    for i, output_detail in enumerate(output_details):
        print(f"Output tensor {i}: shape = {output_detail['shape']}")

    print(f"Processing folder: {args['images_folder']}")

    false_accepts = 0
    false_rejects = 0
    total_accepts = 0
    total_rejects = 0
    total = 0

    num_folders = len([name for name in os.listdir(args['images_folder']) if
                       os.path.isdir(os.path.join(args['images_folder'], name))])

    if num_folders <= 1:
        print(f"WARN: Impossible to check FAR, need more samples (unique face folders)")

    for subdir, dirs, files in os.walk(args['images_folder']):
        for i, dirr in enumerate(dirs):
            far_test_dir_idx = -1
            if num_folders > 1:
                far_test_dir_idx = random.randint(0, num_folders - 1)

            if num_folders > 1 and i == far_test_dir_idx and i == 0:
                far_test_dir_idx += 1
            elif num_folders > 1 and i == far_test_dir_idx and i == (num_folders - 1):
                far_test_dir_idx -= 1
            elif num_folders > 1 and i == far_test_dir_idx:
                far_test_dir_idx += 1

            far_test_image = None
            if far_test_dir_idx >= 0:
                far_test_image = load_rnd_image_from_folder(os.path.join(subdir, dirs[far_test_dir_idx]))

            for image_pairs in load_images_pairs_from_folder(os.path.join(subdir, dirr)):

                template1 = run_facenet_model(image_pairs[0][1], interpreter, input_details, output_details)
                template2 = run_facenet_model(image_pairs[1][1], interpreter, input_details, output_details)
                dist = np.linalg.norm(template1 - template2)

                if args['print_log']:
                    print(f"Score[{image_pairs[0][0]}, {image_pairs[1][0]}] = {dist}")
                if dist > args['threshold']:
                    false_rejects += 1
                total_rejects += 1

                if far_test_image is not None:
                    far_test_template = run_facenet_model(far_test_image[1], interpreter, input_details, output_details)
                    far_test_dist1 = np.linalg.norm(template1 - far_test_template)
                    far_test_dist2 = np.linalg.norm(template2 - far_test_template)
                    if args['print_log']:
                        print(f"Score[{image_pairs[0][0]}, {far_test_image[0]}] = {far_test_dist1}")
                        print(f"Score[{image_pairs[1][0]}, {far_test_image[0]}] = {far_test_dist2}")

                    if far_test_dist1 <= args['threshold']:
                        false_rejects += 1
                    if far_test_dist2 <= args['threshold']:
                        false_rejects += 1
                    total_accepts += 2

                total += 1

    FAR = int(false_accepts / max(total_accepts, 1) * 100)
    FRR = int(false_rejects / max(total_rejects, 1) * 100)
    print(f"Total Pairs: {total}")
    print(f"False Accept Rate %: {FAR}")
    print(f"False Rejection Rate %: {FRR}")
