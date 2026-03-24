import os

def read_strings(input_file):
    list_of_sample = []
    with open(input_file, 'r') as infile:
        for line in infile:
            word = line.strip()
            if word:  # avoid empty lines
                #quoted_word = f"'{word}'"
                list_of_sample.append(word)
    return list_of_sample

#input_filename = 'Run3_sample.txt'
#input_filename = 'data/2018/sample_request.txt'
#input_filename = 'data/2016/sample_request.txt'
input_filename = 'data/2016APV/sample_request.txt'
dss = read_strings(input_filename)

for ds in dss:
    #os.system("rucio rule add --ask-approval --lifetime 15552000 cms:" + ds + " 1 T2_DE_DESY")
    #os.system('rucio rule add --ask-approval --lifetime 15552000 -d "cms:'+ds+'" --skip-duplicates --rses T2_DE_DESY --comment "I need these ffor the Run-3 part of the 3 top searches"')
    os.system('rucio --legacy add-rule --ask-approval --lifetime 15552000 cms:'+ds+' 1 T2_DE_DESY --comment "I need these for the Run-2 part of the XtoYYprime searches"')

