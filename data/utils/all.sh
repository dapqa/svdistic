docker-machine scp "aws:/home/ubuntu/data/saves/*-inferred.txt" ../saves/
docker-machine scp "aws2:/home/ubuntu/data/saves/*-inferred.txt" ../saves/
docker-machine scp "aws3:/home/ubuntu/data/saves/*-inferred.txt" ../saves/

python3 recorrelate.py
for i in $( ls | grep _guesses ); do
  sort -k2,2n -k1,1n -t"," -o $i.resorted $i
  rm $i
done 
python score_only.py
rm *.resorted

