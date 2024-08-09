import math

'''Le coordinate sferiche si calcolano come segue:
raggio (rho) = sqrt(x^2+y^2+z^2)
angolo azimutale (theta θ) = arctan(y/x)
angolo zenitale o polare (phi)= arccos(z/rho) 
'''
def coordinate_sferiche(x,y,z):
    rho = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)
    phi = math.acos(z/rho)
    return rho, theta, phi



#r,t,p=coordinate_sferiche(2,4,5)
#print(r,t,p)