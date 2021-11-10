destination_path="/home/stud/paul/dataset/tunnel/"
dataset_path="/home/stud/paul/dataset/MUT/"

for dir in *; do #für jeden Ordner im aktuellen Verzeichnis
  #move alle im Ordner enthaltenen Daten
  mv /home/stud/paul/dataset/MUT/$dir/* /home/stud/paul/dataset/tunnel/
done