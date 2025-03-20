import random
import secrets

#C:\Users\evilm\AppData\Local\Programs\Python\Python311\python.exe

def not_random_repeating(_):
    return 1

def not_random_addition(seed):
    seed += 3
    seed %= 100
    return seed

def not_random_multiplication(_):
    if not hasattr(not_random_multiplication, "last_value"):
        not_random_multiplication.last_value = 1

    not_random_multiplication.last_value *= 4
    not_random_multiplication.last_value %= 100
    return not_random_multiplication.last_value

def not_random_power(_):
    if not hasattr(not_random_power, "last_value"):
        not_random_power.last_value = 2

    not_random_power.last_value *= not_random_power.last_value
    not_random_power.last_value %= 100
    return not_random_power.last_value

def pseudo_random_lcg(seed, a=14, c=111, m=2**8):
    return (a * seed + c) % m

def pseudo_random_middle_square(seed):
    squared = seed ** 2
    while(squared > 100):
        int(str(squared)[1:])
        squared = squared // 10
    
    return squared
    
def pseudo_random_mersenne(seed):
    if not hasattr(pseudo_random_mersenne, "seed"):
        not_random_addition.seed = seed
        random.seed(not_random_addition.seed)
        
    return random.randint(1, 100)
    
def pseudo_random_prng(_):
    return secrets.randbelow(100)

def main():
    print(not_random_repeating())
    print(not_random_addition())
    print(not_random_multiplication())
    print(not_random_power())
    print(pseudo_random_lcg(55))
    print(pseudo_random_middle_square(55))
    print(pseudo_random_mersenne(55))
    print(pseudo_random_prng())
    
if __name__ == "__main__":
    main()