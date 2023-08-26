from scipy.optimize import linprog

#==========================================================================================================
#Q1: Formulate and solve a linear program to determine the portfolio of stocks,
#bonds, and options that maximises expected profit.
#==========================================================================================================

c = [-12/3, -500/3, 500/3, 10]
A = [[20, 1000, -1000, 90], 
     [0, 1, 1, 0],
     [1, 1000, -1000, 0],
     [1, 0, 0, 1]]
b = [20000, 50, 5000, 1000]
bounds = [(0, None), (0, 50), (0, 50), (0, None)]
from scipy.optimize import linprog
res = linprog(c, A_ub = A, b_ub=b, bounds = bounds)
print(res)
# print results
print("Optimal solution:")
print("x =", round(res.x[0], 2))
print("y =", round(res.x[1], 2))
print("z =", round(res.x[2], 2))
print("b =", round(res.x[3], 2))
print("Expected profit: £", round(-res.fun, 2))
#------------------------------------------------------
#--------------   Results Obtained   ------------------
#
#Optimal solution:
#x = 1000.0
#y = 25.0
#z = 25.0
#b = 111.11
#Expected profit: £ 4000.0
#
#
#-------------------------------------------------------
#==========================================================================================================
#Q2:  Suppose that the investor wants a profit of at least £2,000 in any of the three
#scenarios for the price of XYZ six months from today. Formulate and solve a linear program that will
#maximise the investor’s expected profit under this additional constraint.
#==========================================================================================================
c = [-12/3, -500/3, 500/3, 10]
A = [[20, 1000, -1000, 90], 
     [0, 1, 1, 0],
     [1, 1000, -1000, 0],
     [1, 0, 0, 1],
     [20, -1000, 1000, 90],
     [-8, 1000, -1000, 90],
     [20, -1000, -1000, 90]]
b = [20000, 50, 5000, 1000, 2000, 2000, 2000]
bounds = [(0, None), (0, 50), (0, 50), (0, None)]
from scipy.optimize import linprog
res = linprog(c, A_ub = A, b_ub=b, bounds = bounds)
print(res)
# print results
print("Optimal solution:")
print("x =", round(res.x[0], 2))
print("y =", round(res.x[1], 2))
print("z =", round(res.x[2], 2))
print("b =", round(res.x[3], 2))
print("Expected profit: £", round(-res.fun, 2))#------------------------------------------------------
#--------------   Results Obtained   ------------------
#
#Optimal solution:
#x = 333.33
#y = 27.33
#z = 22.67
#b = 0.0
#Expected profit: £ 2111.11.0
#
#
#-------------------------------------------------------
