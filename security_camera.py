import cv2
import numpy as np
import datetime


def detect_motion_and_record(motion_thresh, record_dur, save_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    recording = False
    record_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, motion_thresh, 255, cv2.THRESH_BINARY)
        motion = np.sum(thresh)

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        thickness = 2
        text = f'Motion: {motion}'

        # Put text on frame
        cv2.putText(frame, text, (10, 30), font, font_scale, font_color, thickness)

        if motion > 0 and not recording:
            recording = True
            record_start_time = datetime.datetime.now()
            out = cv2.VideoWriter(f'{save_path}/motion_{record_start_time.strftime("%Y%m%d_%H%M%S")}.avi',
                                  cv2.VideoWriter_fourcc(*'XVID'), 20.0, size)
            print("Hareket algılandı. Kayıt başlatıldı.")

        if recording:
            out.write(frame)
            elapsed_time = (datetime.datetime.now() - record_start_time).seconds
            if elapsed_time >= record_dur:
                recording = False
                out.release()
                print("Kayıt sona erdi.")

        cv2.imshow('Camera', frame)
        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if recording:
        out.release()
    cv2.destroyAllWindows()


# Input values from user
motion_threshold = int(input("Hareket eşik değeri girin (örn: 100): "))
record_duration = int(input("Kayıt süresi girin (saniye) (örn: 10): "))
save_directory = input("Videonun kaydedileceği dizini girin: ")

detect_motion_and_record(motion_threshold, record_duration, save_directory)
