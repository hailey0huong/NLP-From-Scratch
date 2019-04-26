import math, random
import re
from math import log, exp
from collections import Counter

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''

    #add padding
    s = start_pad(n)+text
    
    #Split the string into a list
    chars = list(s)

    #Generate n-grams
    ngrams = [(''.join(chars[i:i+n]), chars[i+n]) for i in range(len(chars)-n)]
   
    return ngrams


def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n #n-gram
        self.k = k #smoothing
        self.vocab = set()
        self.vocab_size = 0
        self.ngram = []
        self.all_context = []
        self.sorted_vocab = 0
        self.count_ngrams = {}
        self.count_context = {}
        self.text = ''


    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        #save the text
        self.text = text

        #Update ngram list
        new_ngrams = ngrams(self.n, text)
        self.ngram.extend(new_ngrams)

        #print('finish updating ngram')

        #Update vocab (set of all characters used by this model)
        new_vocab= [new_ngrams[i][1] for i in range(len(new_ngrams))]
        self.vocab.update(new_vocab)
        self.sorted_vocab = sorted(list(self.vocab))
        #print('finish updating vocab')
        

        #Update count of vocab
        self.vocab_size = len(self.vocab)
        #print('Vocab size is ',self.vocab_size)

        #Update context list
        new_context = [new_ngrams[i][0] for i in range(len(new_ngrams))]
        self.all_context.extend(new_context)
        #print ('finish updating context')

        #Update count_ngrams
        self.count_ngrams = Counter(self.ngram)
        
        #Update count_context
        self.count_context = Counter(self.all_context)






    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        #Get count of context with char
        count_num = self.count_ngrams[(context, char)] + self.k

        #Get count of context
        count_deno = self.count_context[context] + self.k*self.vocab_size
        if count_deno == 0:
            return 1/self.vocab_size

        #Get the probability
        prob_char = count_num/count_deno

        #print (prob_char)
        return prob_char

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        
        sum_prob_before = 0
        
        for i in range(self.vocab_size):
            if i > 0:
                sum_prob_before += self.prob(context, self.sorted_vocab[i-1])
            sum_prob_after = sum_prob_before + self.prob(context, self.sorted_vocab[i])
            if sum_prob_before <= r and sum_prob_after >r:
                return self.sorted_vocab[i]
        
        return ''
 


    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        if length <= 0:
            return ''
        context = start_pad(length)
        
        for i in range(length):
            char = self.random_char(context[-self.n::])
            context = context + char
            context = context[1:]
            #if i%10 == 0:
                #print ('generating char #{}'.format(str(i)))
                #print (context)
        
        print(context)
        return context



    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        sum_log_perplex = 0
        length_text = len(text)
        ngrams_list = ngrams(self.n, text)
        count_ngrams_dict = Counter(ngrams_list)
        ngrams_unique = list(count_ngrams_dict.keys())
        #print ('finish ngrams...')
        #print ('number of ngrams:', len(ngrams_unique))

        for i in range(len(ngrams_unique)):
            temp = ngrams_unique[i]
            #if i%5000 == 0:
                #print ('Computing probability for ngram #{}'.format(str(i)))
            
            proba = self.prob(temp[0],temp[1])
            if proba == 0.0:
                return float('inf')
            
            sum_log_perplex += count_ngrams_dict[temp]*log(proba)
        
        return exp(-sum_log_perplex/length_text)





        

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################


class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n #n-gram
        self.k = k #smoothing
        self.vocab = set()
        self.vocab_size = 0
        self.ngram = []
        self.all_context = []
        self.sorted_vocab = 0
        self.lambda_list = []
        self.count_ngrams = {}
        self.count_context = {}
        self.text = ''

    def get_vocab(self):
        return self.vocab

    def update(self, text):
        #update text
        self.text = text
        
        for i in range(self.n+1):
            #Update ngram list
            new_ngrams = ngrams(i, text)
            self.ngram.extend(new_ngrams)

            #Update vocab (set of all characters used by this model)
            new_vocab= [new_ngrams[i][1] for i in range(len(new_ngrams))]
            self.vocab.update(new_vocab)
            self.sorted_vocab = sorted(list(self.vocab))

            #Update context list
            new_context = [new_ngrams[i][0] for i in range(len(new_ngrams))]
            self.all_context.extend(new_context)
        
        #Update count of vocab
        self.vocab_size = len(self.vocab)
        #print('Finish training the model')

        #Update count_ngrams
        self.count_ngrams = Counter(self.ngram)
        
        #Update count_context
        self.count_context = Counter(self.all_context)


    def set_lambdas(self, lambdas_list):
        if sum(lambdas_list)>1:
            raise Exception('Sum of the lambdas needs to be equal 1!')
        
        self.lambda_list = lambdas_list

    def prob(self, context, char):
        '''
        lambda_list: lambda values, going from largest ngram to ngram = 0
        '''

        if len(self.lambda_list) ==0:
            self.lambda_list = [1/(self.n+1)]*(self.n + 1)

        #ngram = 0:
        prob_char = self.lambda_list[-1]*(self.count_ngrams[('',char)]+self.k)/(self.count_context['']+ self.k*self.vocab_size)
        
        #ngram >0:
        for i in range(self.n):
            new_context = context[-self.n+i::]
            #Get count of context with char
            count_num = self.lambda_list[i]*(self.count_ngrams[(new_context, char)] + self.k)
   
            #Get count of context
            count_deno = self.count_context[new_context] + self.k*self.vocab_size

            #Get the probability
            prob_char += count_num/count_deno

        #print (prob_char)
        return prob_char




################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    '''
    #Part 0: generating N-grams
    print(ngrams(1, 'abc'))
    print(ngrams(2, 'abc'))
    print ('')

    #Part 1: Create an N-Gram Model
    m = NgramModel(1,0)
    m.update('abab')
    print(m.get_vocab())
    m.update('abcd')
    print(m.get_vocab())
    print(m.prob('a','b'))
    print(m.prob('~','c'))
    print(m.prob('b','c'))
    print ('')

    m = NgramModel(0,0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print([m.random_char('') for i in range(25)])
    print ('')

    m2 = NgramModel(1,0)
    m2.update('abab')
    m2.update('abcd')
    random.seed(1)
    m2.random_text(25)
    print ('')
    '''
    print ('Writing Shakespeare...')
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    m.random_text(250)
    print ('')
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    m.random_text(250)
    print('')
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7, 0.01 )
    m.random_text(250)
    print ('')
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7, 2)
    m.random_text(250)    
    print ('')
    '''
    #Part 2: Perplexity, Smoothing and Interpolation
    print('Perplexity')
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.perplexity('abcd'))
    print(m.perplexity('abca'))
    print(m.perplexity('abcda'))
    print('')

    #Check Perplexity on shakespears file
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', n =7, k = 1)
    print('Perplexity for training text: ',m.perplexity(m.text))
    with open('test_data/nytimes_article.txt', encoding='utf-8', errors='ignore') as f:
        nytimes_text = f.read()
    print('Perplexity for NYtimes article text: ',m.perplexity(nytimes_text))
    with open('test_data/shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        sonnet_text = f.read() 
    print('Perplexity for Shakespeare Sonnets text: ',m.perplexity(sonnet_text))   
     
    print('')

    print('Smoothing')
    m = NgramModel(1, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob('a', 'a'))
    print(m.prob('a', 'b'))
    print(m.prob('c', 'd'))
    print(m.prob('d','a'))
    print('')

    print('Interpolation')
    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    print(m.prob('a', 'a'))
    print(m.prob('a', 'b'))
    m = NgramModelWithInterpolation(2, 1)
    m.update('abab')
    m.update('abcd')
    print(m.prob('~a', 'b'))
    print(m.prob('ba', 'b'))
    print(m.prob('~c', 'd'))
    print(m.prob('bc', 'd'))
    print ('')
    
    print ('Test Perplexity on Interpolation method...')
    print ('Keeping equal weights on lambdas...')
    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', n =7, k = 1)
    print ('k = ', m.k)
    print('Perplexity for training text: ',m.perplexity(m.text))
    print('Perplexity for NYtimes article text: ',m.perplexity(nytimes_text))
    print('Perplexity for Shakespeare Sonnets text: ',m.perplexity(sonnet_text))   
    print ('')

    print ('Keeping equal weights on lambdas...')
    print ('Trying different k...')
    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', n =7, k = 0.5)
    print ('k = ', m.k)
    print('Perplexity for training text: ',m.perplexity(m.text))
    print('Perplexity for NYtimes article text: ',m.perplexity(nytimes_text))
    print('Perplexity for Shakespeare Sonnets text: ',m.perplexity(sonnet_text))   
    print ('')


    print('Adjusting different weights on lambdas...') 
    lambda_list = [0.1,0.1,0.2,0.2,0.1,0.1,0.1,0.1]
    print (lambda_list)
    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', n =7, k = 1)
    print('k = ', m.k)
    m.set_lambdas(lambda_list)
    print('Perplexity for training text: ',m.perplexity(m.text))
    print('Perplexity for NYtimes article text: ',m.perplexity(nytimes_text))
    print('Perplexity for Shakespeare Sonnets text: ',m.perplexity(sonnet_text))   

    print ('') 
    print('Trying different k...') 
    lambda_list = [0.1,0.1,0.2,0.2,0.1,0.1,0.1,0.1]
    print (lambda_list)
    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', n =7, k = 0.01)
    print ('k = ', m.k)
    m.set_lambdas(lambda_list)
    print('Perplexity for training text: ',m.perplexity(m.text))
    print('Perplexity for NYtimes article text: ',m.perplexity(nytimes_text))
    print('Perplexity for Shakespeare Sonnets text: ',m.perplexity(sonnet_text))      
    
    '''
