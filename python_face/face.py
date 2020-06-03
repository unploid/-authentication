import dlib
from skimage import io
from scipy.spatial import distance

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

img = io.imread('me_old.jpg')

win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)

face_descriptor1 = facerec.compute_face_descriptor(img, shape)


print(face_descriptor1)


img = io.imread('me_new.jpg')
win2 = dlib.image_window()
win2.clear_overlay()
win2.set_image(img)
dets_webcam = detector(img, 1)
for k, d in enumerate(dets_webcam):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win2.clear_overlay()
    win2.add_overlay(d)
    win2.add_overlay(shape)

face_descriptor2 = facerec.compute_face_descriptor(img, shape)


a = distance.euclidean(face_descriptor1, face_descriptor2)
print(a)

if a < 0.6:
    print('Most likely there is one person')
else:
    print('Most likely there are two different people')