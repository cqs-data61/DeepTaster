
#!/bin/sh
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
mkdir ./tiny-imagenet-200/val2
cd ./tiny-imagenet-200/train
rm -r n12*
rm -r n09*
rm -r n07*
rm -r n06*
rm -r n04*
rm -r n039*
rm -r n038*
rm -r n037*
rm -r n036*
rm -r n035*
rm -r n034*
rm -r n033*
rm -r n03255*

find ./ -type d -exec mkdir -p "$1/tiny-imagenet-200/val2/{}" \;
cd ..
cd ..
python generate_100classes.py
mv ./tiny-imagenet-200/val ./tiny-imagenet-200/val_original
mv ./tiny-imagenet-200/val2 ./tiny-imagenet-200/val
