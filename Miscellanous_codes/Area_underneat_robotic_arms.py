
# ---------------------------------------------------------------------------------------------------------------- #

# Codice per calcolare l'area sotto al robotic arm deployed al variare degli angoli di deploy presi rispetto 
# all'orizzontale

# ---------------------------------------------------------------------------------------------------------------- #

import sympy as sy


l1,l2,alfa,beta,gamma,delta = sy.symbols('l1 l2 alfa beta gamma delta')

Area = ((l1**2)/2)*(sy.sin(alfa)*sy.cos(alfa)) + ((l1**2)/2)*sy.cos(beta)*(2*sy.sin(alfa) + sy.sin(beta)) + ((l1**2)/2)*sy.cos(gamma)*(2*sy.sin(alfa) + 2*sy.sin(beta)-sy.sin(gamma)) + ((l1*l2)/2)*sy.cos(delta)*(2*sy.sin(alfa) + 2*sy.sin(beta) - 2*sy.sin(gamma) - (l2/l1)*sy.sin(delta)) 
y_dist = (l1*sy.sin(alfa)+l1*sy.sin(beta)-l1*sy.sin(gamma)-l2*sy.sin(delta))

substitution = {

    l1: 0.25,
    l2: 0.16,
    alfa: 60*(sy.pi/180),
    beta: 30*(sy.pi/180),
    gamma: 30*(sy.pi/180),
    delta: 45*(sy.pi/180)

}
# sy.pprint(sy.simplify(Area))
Area_val = Area.xreplace(substitution).evalf()
y_dist_val = y_dist.xreplace(substitution).evalf()

sy.pprint(Area_val)
sy.pprint(y_dist_val)