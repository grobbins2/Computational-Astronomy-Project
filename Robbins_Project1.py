#Code by Grady Robbins                                                                                                                       
#Helped Bella M.,Leah M. after finishing                                                            
import numpy as np
import random, string
import matplotlib.pyplot as plt

data = np.load('wp.npz') #importing War and Peace as basic file
text=data['text']
def calculate_prob(x): #defining probability function for direct letter ordering
    P_temp = []
    for n in range(0,len(x)-1):
        P_temp.append(prob_matrix[number_key[x[n]],number_key[x[n+1]]])
    return sum(P_temp) #/len(P_temp)
def calculate_prob2(x): # defining probability function for letter ordering spaced by 1
    P_temp = []
    for n in range(0,len(x)-1):
        P_temp.append(prob_matrix2[number_key[x[n]],number_key[x[n+1]]])
    return sum(P_temp) #/len(P_temp)
def calculate_prob3(x): # defining probability function for letter ordering spaced by 2
    P_temp = []
    for n in range(0,len(x)-1):
        P_temp.append(prob_matrix3[number_key[x[n]],number_key[x[n+1]]])
    return sum(P_temp) #/len(P_temp)     

def calculate_acc(x): #defining accuracy for comparing of phrases
    score = 0
    for n in range(0,len(x)):
        if true_phrase[n] == x[n]:
            score +=1
    return score*100/len(x)

letters = 'abcdefghijklmnopqrstuvwxyz'
list_of_letters = list(letters)
number_key = dict(zip(list_of_letters,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])) #creating a number key that transforms - 
# - letters into numbers used for indexing in the future

random_solution = list(letters) #generating random order of letters

for n in range(100):
    i = random.randrange(0,len(random_solution))
    j = random.randrange(0,len(random_solution))
    while i == j:
        i = random.randrange(0,len(random_solution))
        j = random.randrange(0,len(random_solution))                                                                                                 
    random_solution[i], random_solution[j] = random_solution[j], random_solution[i] #creating completely random phrase to compare with code
letter_key = dict(zip(list_of_letters, random_solution))
letter_probability_matrix = np.zeros((26,26),tuple) #creating empty probability matrices
letter_probability_matrix2 = np.zeros((26,26),tuple)
letter_probability_matrix3 = np.zeros((26,26),tuple)
txt_string = text.flatten()[0].decode()
char_list = [*txt_string] #turning text file into a list with individual characters

letter_text = [s for s in char_list if s == 'a' or s == 'b'or s == 'c'or s == 'd'or s == 'e'or s == 'f'or s == 'g'or s =='h'or s == 'i'or s == 'j'or s == 'k'or s == 'l'or s == 'm'or s =='n'or s == 'o'or s == 'p'or s == 'q'or s == 'r'or s == 's'or s == 't'or s == 'u'or s =='v'or s =='w'or s == 'x'or s == 'y'or s == 'z']
# only letters left in text list
for n in range(len(letter_text)-1):
    letter_probability_matrix[number_key[letter_text[n]],number_key[letter_text[n+1]]] += 1 #calculating point-based matrix for direct ordering
prob_matrix = letter_probability_matrix*100 /len(letter_text) # turning to % base

for n in range(len(letter_text)-2):
    letter_probability_matrix2[number_key[letter_text[n]],number_key[letter_text[n+2]]] += 1 #matrix for 1 spaced ordering
prob_matrix2 = letter_probability_matrix2*100 /len(letter_text)

for n in range(len(letter_text)-3):
    letter_probability_matrix3[number_key[letter_text[n]],number_key[letter_text[n+3]]] += 1 #matrix for 2 spaced ordering, ineffective   
prob_matrix3 = letter_probability_matrix3*100 /len(letter_text)

true_phrase = list('jackandjillwentupahilltofetchapailofwater')
jumbled_phrase = list('bizcifkbtrrqmfagvijtrraulmazjivitrulqiamh')
chance_phrase = []

for n in range(0,len(true_phrase)):
    chance_phrase.append(letter_key[true_phrase[n]]) #generating random solution to compare with MCMC 
print(chance_phrase)
print("chance accuracy =",calculate_acc(chance_phrase),"%")
new_solution = list(random_solution)
P_chance = calculate_prob(chance_phrase) + calculate_prob2(chance_phrase)  # calculating score for random set of letters 
print("chance probability =",P_chance)
N = 50 #number of loops
probability_graph = []
accuracy_graph = []
for k in range(1,N+1):

    i = random.randrange(0,len(new_solution))
    j = random.randrange(0,len(new_solution))
    while i == j:
        i = random.randrange(0,len(new_solution))
        j = random.randrange(0,len(new_solution)) #generating random, non-equal indices
    
    old_solution = list(new_solution)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i] #commence swap
    old_letter_key = dict(zip(string.ascii_lowercase, old_solution))
    letter_key = dict(zip(string.ascii_lowercase, new_solution))
    new_phrase = []
    for n in range(len(jumbled_phrase)):
        new_phrase.append(letter_key[jumbled_phrase[n]])  #calculating new phrases
    old_phrase = []
    for n in range(len(jumbled_phrase)):
        old_phrase.append(old_letter_key[jumbled_phrase[n]])
    P = (calculate_prob(old_phrase) + calculate_prob2(old_phrase))  #calculating probabilites of phrases using matrixes
    convergence = 1/(np.exp(1/k)) #convergence/Annealing factor
    Pnew = (calculate_prob(new_phrase) + calculate_prob2(new_phrase) )
    
    
    if Pnew < convergence*P:                                                        
        #if new solution is worse than old soluting (with some leeway) then undo swap
        new_solution = list(old_solution)
        new_phrase = list(old_phrase)
        Pnew = P
    probability_graph.append(Pnew)
    accuracy_graph.append(calculate_acc(old_phrase))
accuracy_new = calculate_acc(old_phrase)

print(new_phrase, Pnew)
print("end, the accuracy of new phrase is",accuracy_new,"%")

plt.plot(np.arange(0,N),accuracy_graph) #plotting accuracy and probability over timestep
plt.plot(np.arange(0,N),probability_graph)
plt.xlabel('timestep')
plt.ylabel('accuracy% (blue), probability% (orange)')
plt.savefig('e_annealing1spacing.png')

