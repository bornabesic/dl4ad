import sys
import numpy as np



db_path = "PerceptionCarDataset/"
fc_path = "front/center/"
fl_path = "front/left/"
fr_path = "front/right/"
bc_path = "back/center/"
bl_path = "back/left/"
br_path = "back/right/"

# Get Number of samples and take 97% of it, because each camera has taken a various amount of pictures
# samples = len(open(db_path+fc_path+"poses.txt").readlines())
# samples = int(samples *0.97)

# Open all poses.txt-files for reading
f_fc = open(db_path+fc_path+"poses.txt")
f_fl = open(db_path+fl_path+"poses.txt")
f_fr = open(db_path+fr_path+"poses.txt")
f_bc = open(db_path+bc_path+"poses.txt")
f_bl = open(db_path+bl_path+"poses.txt")
f_br = open(db_path+br_path+"poses.txt")

# Open files for train, and validation set
f_train = open(db_path+"train.txt","w")
f_valid = open(db_path+"validation.txt","w")
f_test= open(db_path+"test.txt","w")

# Just for testing because of missing write permissions on the dataset-directory
#f_train = open("/home/stillerf/virtpython/dl4ad/visual-localization/train.txt","w")
#f_valid = open("/home/stillerf/virtpython/dl4ad/visual-localization/validation.txt","w")


# for counter in range(samples):
lines_fc = f_fc.readlines()
lines_fl = f_fl.readlines()
lines_fr = f_fr.readlines()
lines_bc = f_bc.readlines()
lines_bl = f_bl.readlines()
lines_br = f_br.readlines()

zipped = list(zip(lines_fc, lines_fl, lines_fr, lines_bc, lines_bl, lines_br))

# Create indices for training and validation set inclusive shuffle
samples = len(zipped)
all_indices = np.arange(samples)
np.random.shuffle(all_indices)
# Split 70/15/15
last_train_idx = int(samples * 0.7)
last_valid_idx = int(samples * 0.85)
train_indices = all_indices[:last_train_idx]
valid_indices = all_indices[last_train_idx:last_valid_idx]
test_indices = all_indices[last_valid_idx:]

for i, (line_fc, line_fl, line_fr, line_bc, line_bl, line_br) in enumerate(zipped):
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

    first_element = True
    for element in elements_fc:
        if first_element:
            line_new += ","+fc_path + element 
            first_element = False
        else:
            line_new += ","+element

        
        
    
    #,bl_path+ elements_bl[0], br_path+elements_br[0], *elements_fc[1:]))
  
    if i in train_indices:
        f_train.write(line_new)
    elif i in valid_indices:
        f_valid.write(line_new)
    else:
        f_test.write(line_new)
print("PerceptionCarDataset splitted into training, validation and test set successfully!")
sys.exit()




