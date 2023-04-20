import numpy as np
from lab03.regression import multi_regress


def main(): 
    x= np.linspace (-5,5)
    y= x**3-2*x+8
    Z = np.vstack([np.ones_like(x),x,x**2,x**3]).T
    a,e,rsq = multi_regress(y,Z)
    print(a)
    print(e)
    print(rsq)







if __name__ == "__main__":
    main()