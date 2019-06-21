import cv2

url = 'rtsp://172.16.1.101:5540/ch0'

vcap = cv2.VideoCapture(url)

while(1):
    ret, frame = vcap.read()

    if not ret:
        print('Exit')
        break

    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)

# import rtsp
# with rtsp.Client(rtsp_server_uri = 'rtsp://172.16.1.101:8080/front') as client:
#     client.preview()
