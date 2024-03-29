# This small example illustrates how to use the remote API
# synchronous mode. The synchronous mode needs to be
# pre-enabled on the server side. You would do this by
# starting the server (e.g. in a child script) with:
#
# simRemoteApi.start(19999,1300,false,true)
#
# But in this example we try to connect on port
# 19997 where there should be a continuous remote API
# server service already running and pre-enabled for
# synchronous mode.
#
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

import math

import cv2
import numpy as np
import vrep

from tape import cannyFilter, colourFilter


def run():
    """Main processing loop """

    vrep.simxFinish(-1)  # just in case, close all opened connections
    clientID = vrep.simxStart(
        "127.0.0.1", 19997, True, True, 5000, 5
    )  # Connect to V-REP
    if clientID != -1:
        print("Connected to remote API server")

        # enable the synchronous mode on the client:
        vrep.simxSynchronous(clientID, True)

        # start the simulation:
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

        # get the camera
        res, cameraLeft = vrep.simxGetObjectHandle(
            clientID, "Camera_Left", vrep.simx_opmode_blocking
        )
        res, cameraRight = vrep.simxGetObjectHandle(
            clientID, "Camera_Right", vrep.simx_opmode_blocking
        )
        res, steerHandle = vrep.simxGetObjectHandle(
            clientID, "steer_joint", vrep.simx_opmode_blocking
        )
        res, motorHandle = vrep.simxGetObjectHandle(
            clientID, "motor_joint", vrep.simx_opmode_blocking
        )

        while True:
            vrep.simxSynchronousTrigger(clientID)
            res, resolution, imageLeft = vrep.simxGetVisionSensorImage(
                clientID, cameraLeft, 0, vrep.simx_opmode_blocking
            )
            res, resolution, imageRight = vrep.simxGetVisionSensorImage(
                clientID, cameraRight, 0, vrep.simx_opmode_blocking
            )
            imgL = np.array(imageLeft, dtype=np.uint8)
            imgL.resize([resolution[1], resolution[0], 3])
            imgL = np.flip(imgL, axis=0)

            imgR = np.array(imageRight, dtype=np.uint8)
            imgR.resize([resolution[1], resolution[0], 3])
            imgR = np.flip(imgR, axis=0)

            img = np.concatenate((imgL, imgR), axis=1)

            if res == vrep.simx_return_ok:
                # Process the image

                cannyMasked = cannyFilter(img)
                processed = cv2.cvtColor(cannyMasked, cv2.COLOR_GRAY2BGR)
                processed = img & processed
                processed, _, yellowMask, blueMask = colourFilter(processed)

                blueAverageCentroid = None
                yellowAverageCentroid = None
                moments, centroids, contours = [], [], []

                try:
                    contours, _ = cv2.findContours(
                        yellowMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                    )
                    # get rid of small ones
                    contours = list(filter(lambda x: cv2.contourArea(x) > 10, contours))
                    # sort by size
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    moments = list(map(cv2.moments, contours))
                    centroids = list(
                        map(
                            lambda x: (
                                int(x["m10"] / x["m00"]),
                                int(x["m01"] / x["m00"]),
                            ),
                            moments,
                        )
                    )
                    cv2.drawContours(processed, contours, -1, (0, 0, 255), 3)

                    blueAverageCentroid = centroids[0]

                    for centroid in centroids:
                        cv2.circle(processed, centroid, 3, (255, 0, 0), 3)

                except Exception as e:
                    print("Yellow", e)
                    print(
                        f"Moments: {len(moments)}, Centroids: {len(centroids)}, Contours: {len(contours)}"
                    )

                try:
                    contours, _ = cv2.findContours(
                        blueMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
                    )
                    # get rid of small ones
                    contours = list(filter(lambda x: cv2.contourArea(x) > 10, contours))
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    moments = list(map(cv2.moments, contours))
                    centroids = list(
                        map(
                            lambda x: (
                                int(x["m10"] / x["m00"]),
                                int(x["m01"] / x["m00"]),
                            ),
                            moments,
                        )
                    )

                    yellowAverageCentroid = centroids[0]

                    cv2.drawContours(processed, contours, -1, (0, 0, 255), 3)

                    for centroid in centroids:
                        cv2.circle(processed, centroid, 3, (255, 0, 0), 3)
                except Exception as e:
                    print("Blue", e)
                    print(
                        f"Moments: {len(moments)}, Centroids: {len(centroids)}, Contours: {len(contours)}"
                    )

                height = img.shape[0]
                width = img.shape[1]
                steerangle = 0
                if (
                    blueAverageCentroid is not None
                    and yellowAverageCentroid is not None
                ):
                    bx, by = blueAverageCentroid
                    yx, yy = yellowAverageCentroid
                    centroid = (int(bx / 2 + yx / 2), int(by / 2 + yy / 2))
                    cv2.circle(processed, centroid, 5, (255, 255, 0), 3)
                    x, y = centroid
                    offset = x - width / 2
                    far = height - y
                    steerangle = math.atan2(-offset, far)

                elif blueAverageCentroid is None and yellowAverageCentroid is None:
                    steerangle = 0
                elif yellowAverageCentroid is not None:
                    steerangle = math.radians(30)
                elif blueAverageCentroid is not None:
                    steerangle = math.radians(-30)

                cv2.line(
                    processed,
                    (int(width / 2), 0),
                    (int(width / 2), height),
                    (255, 255, 255),
                    2,
                )

                composite = np.concatenate((img, processed), axis=0)
                # tapes = np.concatenate((yellowMask, blueMask), axis=1)
                cv2.imshow("camera", composite)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # Control the simulated robot
                speed = 200
                if abs(math.degrees(steerangle)) > 15:
                    speed = 10

                # set vehicle speed
                vrep.simxSetJointTargetVelocity(
                    clientID, motorHandle, speed, vrep.simx_opmode_blocking
                )
                # set vehicle steering target
                vrep.simxSetJointTargetPosition(
                    clientID, steerHandle, steerangle, vrep.simx_opmode_blocking
                )

        # stop the simulation:
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

        # Now close the connection to V-REP:
        vrep.simxFinish(clientID)
    else:
        print("Failed connecting to remote API server")
    print("Program ended")


if __name__ == "__main__":
    run()
