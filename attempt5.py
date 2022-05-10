import random

def foo(x):
    return 5*x**5+18*x**4+31*x**3-14*x**2+7*x+19

def fitness(x):
    ans = foo(x)

    if ans == 0:
        return 9876
    else:
        return abs(1/ans)

#Generate solutions
solutions= []
for s in range(100):
    solutions.append(random.uniform(0,1000))
#check to see actually produce values
#print(solutions[:5])

##Genetic algorithm
for _ in range(100):

    ranked_solutions = []
    for s in solutions:
        ranked_solutions.append(fitness(s[0]), s)
    ranked_solutions.sort()
    ## reverse to get biggest first then smallest at the end as fitness closest to zero is aim
    ranked_solutions.reverse()
    print(f"Gen {_} best solutions == ")
    print(ranked_solutions[0])

    if ranked_solution[0][0] > 999:
        break
    
    best_solutions = ranked_solutions[:20]
    elements = []
    for s in best_solutions:
        elements.append(s[1],s[0])

    new_gen = []
    for i in range(100):
        e1 = random.choice(elements) * random.uniform(0.99*1.01) # mutate by 2%
        new_gen.append(e1)

    solutions = new_gen