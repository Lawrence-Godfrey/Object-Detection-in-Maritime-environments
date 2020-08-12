import cv2
import pafy

url = 'https://www.youtube.com/watch?v=6wCltcAg5M0'
video = pafy.new(url)
print('Streaming', video.title)

best = video.getbest(preftype="mp4")

stream = cv2.VideoCapture()
stream.open(best.url)

while stream.isOpened():
	available, frame = stream.read()

	if available:
		cv2.imshow('stream', frame)

		if cv2.waitKey(10) == 27:
			break

stream.release()
cv2.destroyAllWindows() 