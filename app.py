import os
import subprocess
from flask import Flask, request, redirect, jsonify
from speaker_identification.train import train_model
from speaker_identification.converter import convert2wav
import base64 as ba
from flask_cors import CORS
import moviepy.editor as moviepy

app = Flask(__name__)

app.env = 'development'
cors = CORS(app, resorces={r'/*': {"origins": '*'}})


UPLOAD_FOLDER = os.path.join(os.getcwd(), "data_set/")


@app.route("/")
def hello():
    return "Live!"

@app.route("/file-upload", methods=['POST'])
async def file_upload():

    print(request)
    print(request.files)
    
    # audiofila = request.files.get("files")
    # if not audiofila:
    #     resp = jsonify({'message' : 'No Audio File'})
    #     resp.status_code = 400
    #     return resp
    if request.method == 'POST':
        files = request.files.getlist("files")
        label = request.form.get("label")

        print(files)
        # files.save(os.path.join(UPLOAD_FOLDER, 'testdata.webm'))
        count = 0
        if not files:
            resp = jsonify({'message' : 'Bad Request'})
            resp.status_code = 400
            return resp
        for file in files:
            path = os.path.join(UPLOAD_FOLDER, '{0}-{1}.webm'.format(label, count))
            targetWav = os.path.join(UPLOAD_FOLDER, '{0}-{1}.wav'.format(label, count))
            file.save(path)
            command = ["ffmpeg", "-i", path, "-vn", targetWav]
            if subprocess.run(command).returncode == 0:
                print ("FFmpeg Script Ran Successfully")
            else:
                print ("There was an error running your FFmpeg script")

            # file.save(os.path.join(UPLOAD_FOLDER, '{0}-{1}.webm'.format(label, count)))
            # clip = moviepy.VideoFileClip(path)
            # clip.audio.write_audiofile('{0}-{1}.wav'.format(label, count))
            count += 1
        resp = jsonify({'message' : 'File successfully uploaded'})
        resp.status_code = 200
        return resp

@app.route("/file-upload-base64", methods=['POST'])
def file_upload_base64():
    # files = request.form.get("files")
    data = request.get_json()
    print(data)
    print(data['audio'])
    decodestr = ba.b64decode(data['audio'])

    # print(data['files'])
    
    
    wav_file = open("test.webm", "wb")
    # decode_string = ba.b64decode(files)
    wav_file.write(decodestr)
    # wav_file.writef(files)
    resp = jsonify({'message' : 'File successfully uploaded'})
    resp.status_code = 200
    return resp

    # if request.method == 'POST':
    #     files = request.files.getlist("files")
    #     if not files:
    #         resp = jsonify({'message' : 'Bad Request'})
    #         resp.status_code = 400
    #         return resp
    #     for file in files:
    #         file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    #     resp = jsonify({'message' : 'File successfully uploaded'})
    #     resp.status_code = 200
    #     return resp

# https://192.168.0.1:3000/enroll
# {
    # label: "ahmet"
    # files: [base64]
# }

@app.route('/enroll', methods=['POST'])
def enroll():
    
    label = request.form.get("label")
    train_files = request.files.getlist("train_files")
    test_files = request.files.getlist("test_files")
    # files = request.files.getlist("files")
    if not all([label, train_files, test_files]):
        resp = jsonify({'message' : 'Bad Request'})
        resp.status_code = 400
        return resp
    
    target = os.path.join(UPLOAD_FOLDER, label)
    if os.path.exists(target):
        resp = jsonify({'message' : 'Label already exists'})
        resp.status_code = 409
        return resp
    
    os.mkdir(target)
    train_path = os.path.join(target, 'train_set/')
    test_path = os.path.join(target, 'test_set/')
    uploads = os.path.join(os.getcwd(), 'uploads/')
    os.mkdir(train_path)
    os.mkdir(test_path)
    counter = 0
    for train_file in train_files:
        rawPath = os.path.join(uploads, '{0}-train-{1}.webm'.format(label, counter))
        wavPath = os.path.join(train_path, '{0}-train-{1}.wav'.format(label, counter))
        train_file.save(rawPath)
        convert2wav(rawPath, wavPath)
        counter += 1
    



    print("train files saved")

    counter = 0
    for test_file in test_files:
        rawPath = os.path.join(uploads, '{0}-test-{1}.webm'.format(label, counter))
        wavPath = os.path.join(test_path, '{0}-test-{1}.wav'.format(label, counter))
        test_file.save(rawPath)
        convert2wav(rawPath, wavPath)
        counter += 1

    print("test files saved")

    # Train and test data
    acc, fm = train_model()

    resp = jsonify({'message' : 'File successfully uploaded',
                    'accuracy' : acc, 'f_measure' : fm})
    resp.status_code = 200
    return resp

if __name__ == '__main__':
    app.run(debug=True)


