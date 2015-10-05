CCTAG_SOFTWARE_PATH="."
IMAGE_PATH=$DATA_PATH"/cctag/bench/images/synthetic/"$1"Crowns/xp"$2"/images/*.png";
echo $IMAGE_PATH
for j in $IMAGE_PATH; do
    $CCTAG_SOFTWARE_PATH/build/$3/src/detection -i $j -n $1 $CCTAG_SOFTWARE_PATH/parameters/param$1.xml -o $3
    echo $CCTAG_SOFTWARE_PATH/build/$3/src/detection -i $j -n $1 $CCTAG_SOFTWARE_PATH/parameters/param$1.xml -o $3
done
