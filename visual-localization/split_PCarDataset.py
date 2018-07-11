import sys
import numpy as np
import cv2

fc_path = "front/center/"
fl_path = "front/left/"
fr_path = "front/right/"
bc_path = "back/center/"
bl_path = "back/left/"
br_path = "back/right/"

for db_path in ["PerceptionCarDataset/", "PerceptionCarDataset2/"]:

    # Open all poses.txt-files for reading
    f_fc = open(db_path+fc_path+"poses.txt")
    f_fl = open(db_path+fl_path+"poses.txt")
    f_fr = open(db_path+fr_path+"poses.txt")
    f_bc = open(db_path+bc_path+"poses.txt")
    f_bl = open(db_path+bl_path+"poses.txt")
    f_br = open(db_path+br_path+"poses.txt")

    # Open files for train, and validation set
    f_train = open(db_path + "train.txt", "w")
    f_valid = open(db_path + "validation.txt", "w")
    f_test = open(db_path + "test.txt", "w")

    # for counter in range(samples):
    lines_fc = f_fc.readlines()
    lines_fl = f_fl.readlines()
    lines_fr = f_fr.readlines()
    lines_bc = f_bc.readlines()
    lines_bl = f_bl.readlines()
    lines_br = f_br.readlines()

    # Close all poses.txt
    f_fc.close()
    f_fl.close()
    f_fr.close()
    f_bc.close()
    f_bl.close()
    f_br.close()

    zipped = list(zip(lines_fc, lines_fl, lines_fr, lines_bc, lines_bl, lines_br))

    delimiter = " "

    for (line_fc, line_fl, line_fr, line_bc, line_bl, line_br) in zipped:
        elements_fc = line_fc.split(" ")
        elements_fl = line_fl.split(" ")
        elements_fr = line_fr.split(" ")
        elements_bc = line_bc.split(" ")
        elements_bl = line_bl.split(" ")
        elements_br = line_br.split(" ")

        line_new = fl_path+ elements_fl[0]
        line_new += ","+fr_path+elements_fr[0]
        line_new += ","+bc_path+ elements_bc[0]
        line_new += ","+bl_path+ elements_bl[0]
        line_new += ","+br_path+ elements_br[0]

        line_new = delimiter.join([
            fc_path + elements_fc[0],
            fl_path + elements_fl[0],
            fr_path + elements_fr[0],
            bc_path + elements_bc[0],
            bl_path + elements_bl[0],
            br_path + elements_br[0]
        ] + elements_fc[1:]
        )

        img = cv2.imread(db_path + fc_path + elements_fc[0])
        cv2.imshow("Image", img)
        key = cv2.waitKey(0)
        if key == ord(" "):
            print("Skipping...")
            continue
        elif key == ord("t"):
            print("TRAIN")
            f_train.write(line_new)
            continue
        elif key == ord("v"):
            print("VALIDATION")
            f_valid.write(line_new)
            continue
        elif key == ord("s"):
            print("TEST")
            f_test.write(line_new)
            continue
        else:
            print("Random...")
            f = np.random.choice([
                f_train,
                f_valid,
                f_test
            ], p = [0.70, 0.15, 0.15])
            f.write(line_new)
    
    print("{} splitted into training, validation and test set successfully!".format(db_path))
    f_train.close()
    f_valid.close()
    f_test.close()
