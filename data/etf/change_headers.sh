#!/usr/bin/bash

for i in `ls *.csv`
do 
	echo $i;
	sed -i 's/Date/DATE/g' $i;
	sed -i 's/Close/CLOSE/g' $i;
	sed -i 's/High/HIGH/g' $i;
	sed -i 's/Low/LOW/g' $i;
	sed -i 's/Open/OPEN/g' $i;
	sed -i 's/Volume/VOLUME/g' $i;
done