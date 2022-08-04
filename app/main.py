from flask import send_from_directory, Response
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import render_template
from url_utils import get_base_url
import os
import torch

import threading
import cv2
import pafy
import random

random.seed(42)


# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12346
base_url = get_base_url(port)

# 'Derry NH USA':cv2.VideoCapture(pafy.new('https://www.youtube.com/watch?v=FBwVLC_R_r8').allstreams[2].url_https)

videos = { 'Jackson Hole Wyoming USA Town Square':cv2.VideoCapture(pafy.new('https://www.youtube.com/watch?v=1EiC9bvVGnk').allstreams[2].url_https),
            'George Washington Bridge Live Stream':cv2.VideoCapture(pafy.new('https://www.youtube.com/watch?v=S1nmRcAklH0').allstreams[2].url_https),
            'Feed I-75 & Taft Hwy':cv2.VideoCapture(pafy.new('https://www.youtube.com/watch?v=ffsC9km5xDY').allstreams[2].url_https),
            'Porte de Saint-Clair Lyon':cv2.VideoCapture(pafy.new('https://www.youtube.com/watch?v=EBhCrTPpdBI').allstreams[2].url_https)}



# Things to cover:
#     1. Receiving and sending files 
#     2. Receiving and making use of phone numbers/emails
#     3. Displaying live feed videos




# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

    
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

model_traffic = torch.hub.load("ultralytics/yolov5", "custom", path = 'best_T.pt', force_reload=True)
model_traffic.conf = .7
model_stanford = torch.hub.load("ultralytics/yolov5", "custom", path = 'best_S.pt', force_reload=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route(f'{base_url}', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    
            
            

    return render_template("index.html", confidences=None, labels=None,
                        old_filename='place_holder.jpg',
                        filename='place_holder.jpg', found = True)


@app.route(f'{base_url}/uploads/<filename>')
def uploaded_file(filename):
    print("FILENAME:" ,filename)
    here = os.getcwd()
    image_path = os.path.join(here, app.config['UPLOAD_FOLDER'], filename)
    results_cars = model_stanford(image_path, size=416)
    results_accidents = model_traffic(image_path, size=416)
    
    def and_syntax(alist):
        if len(alist) == 1:
            alist = "".join(alist)
            return alist
        elif len(alist) == 2:
            alist = " and ".join(alist)
            return alist
        elif len(alist) > 2:
            alist[-1] = "and " + alist[-1]
            alist = ", ".join(alist)
            return alist
        else:
            return

    
    labels = [results_accidents.pandas().xyxy, results_cars.pandas().xyxy]
    ret_image = cv2.imread(image_path)
    format_confidences = []
    label_for_return = []
    for i in labels:
        if len(i) > 0:
                
            if len(list(i[0]['xmin'])) > 0:
                xmin = int(list(i[0]['xmin'])[0])
                ymin = int(list(i[0]['ymin'])[0])
                xmax = int(list(i[0]['xmax'])[0])
                ymax = int(list(i[0]['ymax'])[0])
            
                label = list(i[0]['name'])
                label_for_return.append(str(label[0]))
                confidences = list(i[0]['confidence'])
                for percent in confidences:
                    format_confidences.append(str(round(percent*100)) + '%')
                

                print(label[0], format_confidences)

                cv2.rectangle(ret_image, (xmin, ymin), (xmax, ymax),color=(255,0,0),thickness=2)
                cv2.putText(ret_image, str(label[0] ), (xmin+10, ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), thickness=2)
            
        else:
            # return the 
            found = False
            return render_template('index.html', labels='No cars or accidents', old_filename=filename, filename=filename, found = False)
            

    
        # https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html    
        # https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-tex

    format_confidences = and_syntax(format_confidences)
    
    save_path = os.path.join(here,  app.config['UPLOAD_FOLDER'], str(filename[:-4] + "_annotated.jpg"))
    
    cv2.imwrite(save_path, ret_image)
    
    
    return render_template('index.html', confidences=format_confidences, labels=and_syntax(label_for_return),
                        old_filename=filename,
                        filename=str(filename[:-4]) + "_annotated.jpg", found = True)
    
    
@app.route(f'{base_url}/uploads/<filename>', methods=['GET', 'POST'])
def uploaded_file_post(filename):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))

    return render_template('index.html')

@app.route(f'{base_url}/uploads/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


def target_function(video):

    
    while True:
        success, image = video.read()
        
        results_accidents = model_traffic(image, size=416)
        results_cars = model_stanford(image, size=416)
        # https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

        # labels[0] = accidents info, labels[1] = cars info
        labels = [results_accidents.pandas().xyxy,  results_cars.pandas().xyxy] 
        label_for_return = []

        # https://stackoverflow.com/questions/58293187/opencv-real-time-streaming-video-capture-is-slow-how-to-drop-frames-or-get-sync
        ret_image = image
        for i in labels:
            if len(i) > 0:
                # annotate an image with the information
                # print("TEST:" , list(i[0]['xmin'])[0])
                if len(list(i[0]['xmin'])) > 0:
                    xmin = int(list(i[0]['xmin'])[0])
                    ymin = int(list(i[0]['ymin'])[0])
                    xmax = int(list(i[0]['xmax'])[0])
                    ymax = int(list(i[0]['ymax'])[0])

                    label = list(i[0]['name'])
                    label_for_return.append(str(label[0]))
                    confidences = str(round(list(i[0]['confidence'])[0]) * 100) + '%'
                    print(str(label[0] + confidences))
                    # for percent in confidences:
                    #     format_confidences.append(str(round(percent*100)) + '%')
                    print("Displaying predictions")
                    cv2.rectangle(ret_image, (xmin, ymin), (xmax, ymax),color=(255,0,0))
                    # alter the text to fit within the bounding box
                    cv2.putText(ret_image, str(label[0] + confidences), (xmin+10, ymin+10), cv2.FONT_HERSHEY_SIMPLEX, .3, (36,255,12), 1)
            else:
                pass


        
        (flag, encodedImage) = cv2.imencode(".jpg", ret_image)
        
        # # ensure the frame was successfully encoded
        # # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
    
#background process happening without any refreshing
@app.route('/background_process_george_washington_bridge')
def background_process_test():
    t = threading.Thread(target=target_function(), name="name", args=videos['George Washington Bridge Live Stream'])
    t.daemon = True
    t.start()
    return ("nothing")

@app.route("/video_feed_george_washington_bridge")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(target_function(videos['George Washington Bridge Live Stream']),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


#background process happening without any refreshing
@app.route('/background_process_test_jackson_hole')
def background_process_test_1():
    t = threading.Thread(target=target_function(), name="name", args=videos['Jackson Hole Wyoming USA Town Square'])
    t.daemon = True
    t.start()
    return ("nothing")

@app.route("/video_feed_jackson_hole")
def video_feed_1():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(target_function(videos['Jackson Hole Wyoming USA Town Square']),
		mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc18.ai-camp.dev'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)


# from flask import send_from_directory, Response
# from flask import Flask, flash, request, redirect, url_for
# from werkzeug.utils import secure_filename
# from flask import render_template
# from url_utils import get_base_url
# import os
# import torch


# import threading
# import cv2
# import pafy


# # setup the webserver
# # port may need to be changed if there are multiple flask servers running on same server
# port = 12345
# base_url = get_base_url(port)

# video = cv2.VideoCapture(pafy.new('https://www.youtube.com/watch?v=1EiC9bvVGnk').allstreams[2].url_https)
# # while True:
# #     success, frame = video.read()
# #     print(success)

# # if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
# if base_url == '/':
#     app = Flask(__name__)
# else:
#     app = Flask(__name__, static_url_path=base_url+'static')

# UPLOAD_FOLDER = 'static/uploads'
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# # model_Stanford1 = torch.hub.load("ultralytics/yolov5", "custom", path = 'best-Oliver-S.pt', force_reload=True)
# # model_Traffic1 = torch.hub.load("ultralytics/yolov5", "custom", path = 'best_{ian kengott}_T_.pt', force_reload=True)
# #model_Stanford2 = torch.hub.load("ultralytics/yolov5", "custom", path = 'best_Patrick_S.pt', force_reload=True)
# # model_Traffic2 = torch.hub.load("ultralytics/yolov5", "custom", path = 'best_Moses_S.pt', force_reload=True)

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route(f'{base_url}', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)

#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('uploaded_file',
#                                     filename=filename))
        
#     return render_template('index.html')



# @app.route(f'{base_url}/uploads/<filename>')
# def uploaded_file(filename):
#     here = os.getcwd()
#     image_path = os.path.join(here, app.config['UPLOAD_FOLDER'], filename)
#     results = model(image_path, size=416)
#     if len(results.pandas().xyxy) > 0:
#         results.print()
#         save_dir = os.path.join(here, app.config['UPLOAD_FOLDER'])
#         results.save(save_dir=save_dir)
#         def and_syntax(alist):
#             if len(alist) == 1:
#                 alist = "".join(alist)
#                 return alist
#             elif len(alist) == 2:
#                 alist = " and ".join(alist)
#                 return alist
#             elif len(alist) > 2:
#                 alist[-1] = "and " + alist[-1]
#                 alist = ", ".join(alist)
#                 return alist
#             else:
#                 return
#         confidences = list(results.pandas().xyxy[0]['confidence'])
#         # confidences: rounding and changing to percent, putting in function
#         format_confidences = []
#         for percent in confidences:
#             format_confidences.append(str(round(percent*100)) + '%')
#         format_confidences = and_syntax(format_confidences)

#         labels = list(results.pandas().xyxy[0]['name'])
#         # labels: sorting and capitalizing, putting into function
#         labels = set(labels)
#         labels = [emotion.capitalize() for emotion in labels]
#         labels = and_syntax(labels)
#         return render_template('results.html', confidences=format_confidences, labels=labels,
#                                old_filename=filename,
#                                filename=filename)
#     else:
#         found = False
#         return render_template('results.html', labels='No Emotion', old_filename=filename, filename=filename)


# @app.route(f'{base_url}/uploads/<path:filename>')
# def files(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


# # def target_function(video):
# #     while True:
# #         success, image = video.read()
        
# #         results_accidents = model_Traffic2(image, size=416)
# #         results_cars = model_Stanford1(image, size=416)
# #         # https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
# #         labels_accidents = list(results_accidents.pandas().xyxy[0]['name'])
# #         # if len(labels_accidents) > 0:
# #             # # email/text the user that there is a crash
# #             # print('hell')
        
# #         image = results_accidents.render()[0]

# #         (flag, encodedImage) = cv2.imencode(".jpg", image)
# #         # ensure the frame was successfully encoded
# #         # yield the output frame in the byte format
# #         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
# #                bytearray(encodedImage) + b'\r\n')

# # #background process happening without any refreshing
# # @app.route('/background_process_test')
# # def background_process_test():
# #     t = threading.Thread(target=target_function(), name="name", args=video)
# #     t.daemon = True
# #     t.start()
# #     return ("nothing")

# # @app.route("/video_feed")
# # def video_feed():
# # 	# return the response generated along with the specific media
# # 	# type (mime type)
# # 	return Response(target_function(video),
# # 		mimetype = "multipart/x-mixed-replace; boundary=frame")



# # define additional routes here
# # for example:
# # @app.route(f'{base_url}/team_members')
# # def team_members():
# #     return render_template('team_members.html') # would need to actually make this page

# if __name__ == '__main__':
#     # IMPORTANT: change url to the site where you are editing this file.
#     website_url = 'cocalc18.ai-camp.dev'
    
#     print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
#     app.run(host = '0.0.0.0', port=port, debug=True)
