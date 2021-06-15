# Script for runing text_processing.py and readability_measure.py

conda activate capstone

python3 text_processing.py -f ../data/train.csv -c excerpt -nc cleaned_text -nf cleaned_text.csv

python3 readability_measures.py -i ../data/train.csv -f excerpt -o ../data/outputs/text_readability
