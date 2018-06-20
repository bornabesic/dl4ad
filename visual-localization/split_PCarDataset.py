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
samples = len(open(db_path+fc_path+"poses.txt").readlines())
samples = int(samples *0.97)

# Create indices for training and validation set inclusive shuffle
all_indices = np.arange(samples)
np.random.shuffle(all_indices)
# Split 80/20
splitpoint = int(samples * 0.8)
train_indices = all_indices[:splitpoint]
valid_indices = all_indices[splitpoint:]

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

# Just for testing because of missing write permissions on the dataset-directory
#f_train = open("/home/stillerf/virtpython/dl4ad/visual-localization/train.txt","w")
#f_valid = open("/home/stillerf/virtpython/dl4ad/visual-localization/validation.txt","w")


for counter in range(samples):
    line_fc = f_fc.readline()
    line_fl = f_fl.readline()
    line_fr = f_fr.readline()
    line_bc = f_bc.readline()
    line_bl = f_bl.readline()
    line_br = f_br.readline()

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
  
    
    if counter in valid_indices:
        f_valid.write(line_new)
    else:
        f_train.write(line_new)
print("PerceptionCarDataset splitted into training and validation set successfully!")
sys.exit()




