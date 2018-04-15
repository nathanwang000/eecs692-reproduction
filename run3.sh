#!/bin/bash

# first experiment: 4 hours
# for i in {1..5};
# do
#     run="models/run$i"
#     # STN
#     # paper RTS
#     python main.py -r 45 -t 0.1 -s 0.3 -l $run
#     # paper R only
#     python main.py -r 90 -t 0 -s 0 -l $run

#     # CNN
#     # paper RTS
#     python main.py -m cnn -r 45 -t 0.1 -s 0.3 -l $run
#     # paper R only
#     python main.py -m cnn -r 90 -t 0 -s 0 -l $run
# done

# # paper RTS: give up
# python main.py -m fcn -r 45 -t 0.1 -s 0.3
# # paper R only
# python main.py -m fcn -r 90 -t 0 -s 0

# second experiment: 2 hours
# for i in {1..5};
# do
#     run="models/run$i"
#     python main.py -e double -r 45 -t 0.1 -s 0.3 -l $run
#     # baseline 
#     python main.py -e double -m cnn -r 45 -t 0.1 -s 0.3 -l $run
# done

# third experiment: 7 hours
for i in {1..5};
do
    run="models/run$i"
    # if border helps
    python main.py -e one -i lr -p border -r 45 -t 0.1 -s 0.3 -l $run/border/lr
    python main.py -e one -i lr -p zeros -r 45 -t 0.1 -s 0.3 -l $run/zeros/lr
    # if initialization helps
    python main.py -e one -i id -p border -r 45 -t 0.1 -s 0.3 -l $run/border/identity
    python main.py -e one -i id -p zeros -r 45 -t 0.1 -s 0.3 -l $run/zeros/identity
    python main.py -e one -i rand -p border -r 45 -t 0.1 -s 0.3 -l $run/border/rand
    python main.py -e one -i rand -p zeros -r 45 -t 0.1 -s 0.3 -l $run/zeros/rand
    # baseline
    python main.py -e one -m cnn -r 45 -t 0.1 -s 0.3 -l $run
done
