import os, sys, time, asyncio
from .src.predict_signal import predict_bio_sigs
import math
import cv2

real_path = os.path.dirname(os.path.realpath(__file__))
sub_path = os.path.split(real_path)[0]
os.chdir(sub_path)

from flask import Flask, escape, request,  Response, g, make_response, jsonify
from flask.templating import render_template
from werkzeug.utils import secure_filename

from camera import VideoCamera

app = Flask(__name__)
app.debug = True

video_camera = None
global_frame = None

def root_path():
	'''root 경로 유지'''
	real_path = os.path.dirname(os.path.realpath(__file__))
	sub_path = "\\".join(real_path.split("\\")[:-1])
	return os.chdir(sub_path)

def close_cam():
	global video_camera
	if video_camera:
		del video_camera
		video_camera = None


''' Main page '''
@app.route('/')
def index():
	close_cam()
	return render_template('index.html')

''' About page '''
@app.route('/about')
def about():
	close_cam()
	return render_template('about.html')

''' Team info page '''
@app.route('/team')
def team():
	close_cam()
	return render_template('team.html')

''' Result page '''
@app.route('/result_get')
def result():
	close_cam()
	return render_template('result_get.html')

@app.route('/result_post', methods=['GET', 'POST'])
def result_post():
	if request.method == 'POST':
		root_path()
		# User Video
		user_video_path = 'sj\\static\\video\\video.avi'
		user_video_type = 'mp4'
		result_spo2_hr_rr = {}

		# video file
		if user_video_type in ['avi', 'AVI', 'mp4', 'MP4', 'MPEG', 'mkv', 'MKV']:
			print('Type is Video')
			print(os.path.isfile(user_video_path))
			result_spo2_hr_rr = predict_bio_sigs(user_video_path, 10, [1, 12])
		close_cam()
		return render_template('result_post.html', result=result_spo2_hr_rr, zip=zip)


@app.route('/record_status', methods=['POST'])
def record_status():
	global video_camera
	if video_camera == None:
		video_camera = VideoCamera()

	json = request.get_json()

	status = json['status']

	if status == "true":
		video_camera.start_record()
		return jsonify(result="started")
	else:
		video_camera.stop_record()
		return jsonify(result="stopped")


def video_stream():
	global video_camera
	global global_frame

	if video_camera == None:
		video_camera = VideoCamera()
  
	return video_camera.get_frame()


@app.route('/video_viewer')
def video_viewer():
	return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
 