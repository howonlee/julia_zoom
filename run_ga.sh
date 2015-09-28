for i in `seq 1 1`;
do
  python 2_fractal.py $i > /dev/null &
done
