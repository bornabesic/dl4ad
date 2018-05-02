
dataset:
	wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
	unzip GTSRB_Final_Training_Images.zip -d training
	mv training/GTSRB/Final_Training/Images/* training/
	rm -rf training/GTSRB/
	wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip
	unzip GTSRB_Final_Test_Images.zip -d test
	mv test/GTSRB/Final_Test/Images/* test/
	rm -rf test/GTSRB/
	wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip
	unzip GTSRB_Final_Test_GT.zip -d test
	rm -rf GTSRB_Final_Training_Images.zip
	rm -rf GTSRB_Final_Test_Images.zip
	rm -rf GTSRB_Final_Test_GT.zip

clean_models:
	rm -rf models/
