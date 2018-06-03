
dataset:
	wget http://deeploc.cs.uni-freiburg.de/static/datasets/deeploc.tar.gz
	tar -xvzf deeploc.tar.gz
	./augmentation.py

check_syntax:
	python3 -m py_compile *.py

clean_dataset:
	rm -rf DeepLoc/

clean_augmented:
	rm -rf DeepLocAugmented/
