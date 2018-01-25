from tokenizer import tokenizeSentence as tokenize

with open('text/full.txt', 'r') as text:
    c = 0
    for line in text:
        if c%100 == 0:
            print "iteration num: " + str(c)
        c+=1
        sentence = tokenize(line) + "\n"
        with open ('text/full-tokenized.txt', 'a') as textTokenized:
            textTokenized.write(sentence)

print "it's done"