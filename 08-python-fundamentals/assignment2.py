import math


def case1(k, n):  # order matters, repetition allowed
    c = int(n ** k)
    print(f'\nIf a toy store has {n} kinds of gift sets, one have {c} ways to prepare the presents for his/her {k} friends.')

def case2(k, n): # order matters, no repetition
    c = int(math.factorial(n) / math.factorial(n - k))
    print(f'\nThere are {c} different tickets to buy to win a jackpot of "{k} out of {n}"  ')

def case3(k, n): # order does not matter, no repetition
    c = int(math.factorial(n) / (math.factorial(n - k) * math.factorial(k)))
    print(f'\nA tapas restaurant serves {k} tapas for a ready to share manu and today they offer {n} different kinds of tapas to choose from. So today customers have {c} choices.')

def case4(k, n): # distribute k to n
    c = int(math.factorial(n+k-1)/(math.factorial(k)*math.factorial(n-1)))
    print(f'\nA group of volunteer has built {n} homes for {k} birds in an area, there are {c} different ways for the {k} birds to stay. Note that one home can hold multiple birds at the same time.')

print('Enter the integers k and n separated by comma')
s = input().split(',')
k = int(s[0])
n = int(s[1])

if k <= n:
    case1(k, n)
    case2(k, n)
    case3(k, n)
    case4(k, n)
else:
    print(f'\nk={k} is larger than n={n}, hence no repetition cases are undefined. So we only consider the cases when repetitions are allowed.')
    case1(k, n)
    case4(k, n)