import time

import cv2
import numpy as np
import threading
lock = threading.Lock()
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh


class RecordingThread (threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('./sj/static/video/video.avi',fourcc, 10.0, (640,480))
        
        # self.fps_log = []

    def run(self):
        time_elapsed = 0
        FRAME_RATE = 10
        ptime = time.time()
        while self.isRunning:
            time_elapsed = time.time() - ptime
            if time_elapsed > 1./FRAME_RATE:
                with lock:
                    ret, frame = self.cap.read()
                ctime = time.time()
                if ret:
                    self.out.write(frame)
                    
                # # log fps
                # fps = 1/(ctime - ptime)
                # ptime = ctime
                # self.fps_log.append(fps)
            
        self.out.release()

    def stop(self):
        self.isRunning = False
        # np.savetxt('wFps-Log', np.asarray(self.fps_log), fmt='%2.3f')

    def __del__(self):
        self.out.release()


class VideoCamera(object):
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)

        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None
        
        # mediapipe face mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)
        self.landmarks = {  # landmarks of mesh index
            'left_chick': [116, 203],
            'right_chick': [349, 376],
            # 'forehead': [67, 296],
        }
        
    def __del__(self):
        self.cap.release()
        self.face_mesh.close()
    
    def get_frame(self):
        global_frame = None
        ptime = time.time()
        FRAME_RATE = 5
        cnt = 0
        while True:
            time_elapsed = time.time() - ptime
            if time_elapsed > 1./FRAME_RATE:
                ptime = time.time()
                # frame = self._cap_and_encode()
                with lock:
                    ret, frame = self.cap.read()

                if ret:
                    # draw rectangles using mediapipe face mesh
                    results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            for ltop, rbot in self.landmarks.values():
                                loc_x = int(face_landmarks.landmark[ltop].x * frame.shape[1])
                                loc_y = int(face_landmarks.landmark[ltop].y * frame.shape[0])
                                loc_xx = int(face_landmarks.landmark[rbot].x * frame.shape[1])
                                loc_yy = int(face_landmarks.landmark[rbot].y * frame.shape[0])
                                cv2.rectangle(frame, (loc_x, loc_y), (loc_xx, loc_yy), (0,0,255), 2)
                    
                    # # display fps
                    # ctime = time.time()
                    # fps = 1/(ctime - self.ptime)
                    # self.ptime = ctime
                    if self.is_record:
                        cnt += 1
                        cv2.putText(frame, f'sec: {cnt // FRAME_RATE}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                                    2, (0, 255, 0), 2)
                    else:
                        cnt = 0
                                        
                    # image encode
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    
                    frame = jpeg.tobytes()

                    if frame != None:
                        global_frame = frame
                else:
                    frame = global_frame

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def _cap_and_encode(self):
        with lock:
            ret, frame = self.cap.read()

        if ret:
            
            # draw rectangles using mediapipe face mesh
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for ltop, rbot in self.landmarks.values():
                        loc_x = int(face_landmarks.landmark[ltop].x * frame.shape[1])
                        loc_y = int(face_landmarks.landmark[ltop].y * frame.shape[0])
                        loc_xx = int(face_landmarks.landmark[rbot].x * frame.shape[1])
                        loc_yy = int(face_landmarks.landmark[rbot].y * frame.shape[0])
                        cv2.rectangle(frame, (loc_x, loc_y), (loc_xx, loc_yy), (0,0,255), 2)
            
            # # display fps
            # ctime = time.time()
            # fps = 1/(ctime - self.ptime)
            # self.ptime = ctime
            # cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
            #             3, (0, 255, 0), 2)
            
            # image encode
            ret, jpeg = cv2.imencode('.jpg', frame)
            
            time.sleep(0.1)

            return jpeg.tobytes()
        else:
            return None

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()

            