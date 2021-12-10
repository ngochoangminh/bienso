import random
alphabet  = '0123456789ABCDEFGHIJKLMNOPQRSVTWXYZ'
def gen(n):
    r = ''.join(random.choice(alphabet) for _ in range(n))
    return r+'\n'

with open('train_label.txt', 'w') as f:
    for i in range(1000000):
        f.write(gen(15))