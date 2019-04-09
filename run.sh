#!/bin/bash

echo "Run handin template"

# Script that pipes output of question 1 to a file
echo "Run the first script ..."
python3 qn1.py > qn1.txt

# Script that pipes output of question 1 to a file
echo "Run the second script ..."
python3 qn2.py > qn2.txt

# Script that pipes output of question 1 to a file
echo "Run the third script ..."
python3 qn3.py 

echo "Generating the pdf"
pdflatex report.tex
