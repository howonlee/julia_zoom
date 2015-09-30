for i in `seq 1 500`;
do
  echo $i;
  python 3_fractal.py $i
done
