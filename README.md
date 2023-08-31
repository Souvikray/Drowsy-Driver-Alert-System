# Note from Fork owner [psychoSherlock](https://psychoSherlock.github.io)

This fork contains fix for `index error` running the file.
If you want to create a custom `trainer.yml` file, use [this repo](https://github.com/ITCoders/Human-detection-and-Tracking.git)

- Use `DrowsyDriver.py` if you want do not want the face recognition feature.
- Use `DrowsyDriverApplication.py` if you want the face recognition.

# Note from Author [Souvikrat](https://github.com/souvikray/)

I have build an application that can detect the drowsiness of a driver and alert the driver before any fatal incident occurs.

There are basically three phases in the system

**FACE RECOGNITION**

This section uses the OpenCV Library to perform the operations. Initially, the driver has to take the driving seat. As soon as the system detects a face, it initiates the process of face recognition.So here, certain continuous frames of the driver’s face is used to identify the face of the driver. The system compares the given frame with the saved pictures of the person in the database. If a match is found, the system identifies the face and shows the name of the person on the screen.
The system also saves the identified person’s name and fetches the contacts of that person from the database to be used in case of emergency.

**EYE BLINK DETECTION**

Here DLib library is used to perform the next set of operations. After the face is recognised and the driver’s data is loaded up, now it performs real time eye blink detection.So the system localises the eye area of the person and checks if the person is drowsing. At any sight of drowsing, it immediately alerts the driver by playing an alarm.The driver is expected to wake up at the sound. If for some reason, the driver continues to drowse, it will trigger the next operation.

**FETCH LOCATION AND SEND ALERT**

The next part of the operation involves sending an alert message to the driver’s predefined contacts in case of long duration of inactivity.So the driver’s current location is fetched using an online location service called geocoder.Then an SOS message is prepared which contains a google maps link to the driver’s current location and this message is sent to the predefined contacts using a service called Twilio.

To understand how face recognition and eye blink detection works, check out my repositories below

https://github.com/Souvikray/Realtime-Face-Recognition/blob/master/README.md

https://github.com/Souvikray/Eye-Blink-Detection/blob/master/README.md

**NOTE** I have already trained my system for face recognition.Here I am simply loading up the yml file that contains the pretrained data to speed up the system.

Now let me show you my application in action

**1. Initially the driver takes the wheel**

![Alt text](https://github.com/Souvikray/Drowsy-Driver-Alert-System/blob/master/screenshot1.jpg?raw=true "Optional Title")

**2. The face is detected**

![Alt text](https://github.com/Souvikray/Drowsy-Driver-Alert-System/blob/master/screenshot2.png?raw=true "Optional Title")

**3. The system initiates a face recognition scan**

![Alt text](https://github.com/Souvikray/Drowsy-Driver-Alert-System/blob/master/screenshot3.png?raw=true "Optional Title")

**4. The system localises the eye region**

![Alt text](https://github.com/Souvikray/Drowsy-Driver-Alert-System/blob/master/screenshot4.png?raw=true "Optional Title")

**5. The system starts eye blink detection check**

![Alt text](https://github.com/Souvikray/Drowsy-Driver-Alert-System/blob/master/screenshot5.png?raw=true "Optional Title")

**6. If drowsiness detected, an alert sound is played**

![Alt text](https://github.com/Souvikray/Drowsy-Driver-Alert-System/blob/master/screenshot6.png?raw=true "Optional Title")

**7. If driver is inactive for long duration an emergency alert message is sent to the driver’s contacts**

![Alt text](https://github.com/Souvikray/Drowsy-Driver-Alert-System/blob/master/screenshot7.png?raw=true "Optional Title")

Here is a short gif showing the system in action

<a href="https://imgflip.com/gif/2anma4"><img src="https://i.imgflip.com/2anma4.gif" title="made at imgflip.com"/></a>
