import sys
import numpy as np
import cv2
import os
from gelsight import gsdevice
from gelsight import gs3drecon


"""
xmlrpc
"""
import xmlrpc.client
proxy = xmlrpc.client.ServerProxy("http://192.168.40.216:9120")
# proxy.move(0,0,0,False)
# proxy.move(50,0,0,False)



def main(argv):
    # Set flags
    SAVE_VIDEO_FLAG = False
    FIND_ROI = False
    GPU = False
    # GPU = True
    MASK_MARKERS_FLAG = False

    # Path to 3d model
    path  = os.path.dirname(os.path.realpath(__file__))

    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gsdevice.Camera("GelSight Mini")
    net_file_path = 'nnmini.pt'

    dev.connect()

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(dev)
    net = nn.load_nn(net_path)

    f0 = dev.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1], f0.shape[0]), isColor=True)
        print(f'Saving video to {file_path}')

    if FIND_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('roi = ', roi)
    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    # vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)



    count = 0
    data = (0,0,0,0,0,0)
    data_list = list(data)
    data_list[0] = count
    data = tuple(data_list)

    flag = False

    f1 = dev.get_image()
    bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
    lastFrame = bigframe
    try:
        import time
        t1 = time.time()
        while dev.while_condition:
            t2 = time.time()
            print(t2 - t1)
            t1 = t2
            # get the roi image
            f1 = dev.get_image()
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))

            # diff = max(max(abs(bigframe[:,:,2] - lastFrame[:,:,2])))
            # diff = np.max(cv2.absdiff(bigframe,lastFrame))
            # print(f"diff = {diff}")
            # cv2.imshow('Image', bigframe)
            
            

            f1_gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)

            ''' Display the results '''
            # vis3d.update(dm)

            # print(f'Max height value is {np.max(dm)}')


            dm_2 = dm*0.06
            bigdm = cv2.resize(dm_2, (f1.shape[1] * 2, f1.shape[0] * 2))
            # dm_3 = np.zeros_like(dm)
            dm_3 = dm.copy()
            # dm_3 = bigframe.copy()
            dm_3[dm < 1.3] = 0

            # # dm_diff = np.abs(dm_3 - dm_prev)
            # dm_diff = dm_3 - dm_prev
            # # dm_diff = bigframe - dm_prev
            # # dm_diff = f1_gray - dm_prev
            # dm_diff[dm_diff < 2] = 0
            # dm_diff[dm_diff > 2] = 255
            # dm_prev = dm_3
            # # dm_prev = bigframe
            # # dm_prev = f1_gray

            # total_force = np.sum(dm_3/(320*240))
            total_force = np.mean(dm_3)
            if total_force < 0.03:
                total_force = 0
            max_force = np.max(dm_3)
            if max_force < 0.3:
                max_force = 0
            if total_force < 0.02:
                cv2.putText(bigframe, "Mean: {:.0f}".format(total_force), (100,50), 20, 1.0, (0,255,0), 2)
            else:
                cv2.putText(bigframe, "Mean: {:.2f}".format(total_force), (100,50), 20, 1.0, (0,255,0), 2)
            if max_force < 0.05:
                cv2.putText(bigframe, "Max: {:.0f}".format(max_force), (100,100), 20, 1.0, (0,255,0), 2)
            else:
                cv2.putText(bigframe, "Max: {:.2f}".format(max_force), (100,100), 20, 1.0, (0,255,0), 2)
            cv2.imshow('Image', bigframe)
            cv2.imshow('Height Map', bigdm)
            # cv2.imshow('Height Diff', dm_diff)
            # cv2.waitKey(10)

            data_list = list(data)
            data_list[0] = round(total_force, 2)
            data_list[1] = round(max_force, 2)
            data = tuple(data_list)
            # udp.send((str(data)+'\n').encode('utf-8'), addr)
            maxData = data[1]
            
            inputKey = cv2.waitKey(1)
            if flag == True and  nn.dm_zero_counter >= 50 and maxData > 2:
            # if flag == True and  nn.dm_zero_counter >= 50 :
                proxy.stop()
                flag = False
                pass
            if inputKey& 0xFF == ord('q'):
                break
            elif inputKey & 0xFF == ord('w'):
                print("wwww")
                proxy.move(0,0,0,False)
                flag = False
            elif inputKey & 0xFF == ord('e'):
                print("eeee")
                proxy.move(50,0,0,False)
                flag = True
            elif inputKey & 0xFF == ord('s'):
                proxy.stop()
                flag = False
                pass
            # lastFrame = bigframe
            # if diff > 20 and flag == True:
            #     proxy.stop()
            #     flag = False
            #     pass

            if SAVE_VIDEO_FLAG:
                out.write(f1)
            # print(proxy.position())

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.stop_video()


if __name__ == "__main__":
    main(sys.argv[1:])
