import os, sys
import glob

files = glob.glob('*.jpg')
for file in files:
	original_name = file
	if original_name.find("-small") > -1:
		new_name = original_name[:-10] + ".jpg"
		print (new_name)
		os.rename(original_name, new_name)
	else:
		print ("File name already named.")
print ("Renaming finished!")
