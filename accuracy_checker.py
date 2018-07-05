'''
Created on Sep 30, 2015

@author: Philip Schulz
'''

import sys

def main(args):
    
    gold_labels = dict()
    pred_labels = dict()

    try:
        with open(args[0]) as goldFile:
            for line in goldFile:
                elements = line.split()
                gold_labels[elements[0]] = elements[1]
                
        with open(args[1]) as pred_file:
            for line in pred_file:
                elements = line.split()
                pred_labels[elements[0]] = elements[1]
    
    except IOError as e:
        print(e)
        print("One of the files does not exist on your computer.")
        sys.exit(0)
        
    if len(gold_labels) != len(pred_labels):
        print ('The lists are of different size. Please make sure to only ' +
        'use equally sized lists.')
        sys.exit(0)
        
    overlap = 0
    for doc, label in gold_labels.items():
        if pred_labels[doc] == label:
            overlap += 1
    
    overlap /= float(len(gold_labels))
    overlap *= 100
    
    print('The overlap between the gold list and the student ouput is {}%'.format(overlap))

if __name__ == '__main__':
    main(sys.argv[1:])
