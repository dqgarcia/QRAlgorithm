# attempt at Burnden and Fairies eigenvalue approximation for symmetric matrices
# input symmetric matrix
# use householder's transformation to convert into tridiagonal form
# then use QR algorithm to find eigenvalues

import numpy

n = 4                                   # dimension
A = [[4.0,  1.0,    -2.0,   2.0],       # matrix
    [1.0,   2.0,    0.0,    1.0],
    [-2.0,  0.0,    3.0,    -2.0],
    [2.0,   1.0,    -2.0,   -1.0]]


# Householder's transformation


for k in range (0, n-2):
    q = 0
    alpha = 0
    v = []
    u = []
    z = []
    for j in range (0,n):
        v.append(0.0)
        u.append(0.0)
        z.append(0.0)

    for j in range (k+1, n):
        q += A[j][k] * A[j][k]
    
    if A[k+1][k] == 0:
        alpha = -numpy.sqrt(q)
    else:
        alpha = -((numpy.sqrt(q) * A[k+1][k])/(numpy.absolute(A[k+1][k])))
    
    RSQ = (alpha * alpha) - (alpha * A[k+1][k])


    v[k+1] = A[k+1][k] - alpha

    for j in range (k+2, n):
        v[j] = A[j][k]
    
    for j in range (k, n):
        for i in range (k+1, n):
            u[j] += A[j][i] * v[i]
        u[j] *= (1/RSQ)

    prod = 0
    for i in range (k+1, n):
        prod += v[i] * u[i]
    
    for j in range (k, n):
        z[j] = u[j] - ((prod/(2 * RSQ)) * v[j])
    
    for l in range (k+1, n-1):
        for j in range (l+1, n):
            A[j][l] = A[j][l] - (v[l] * z[j]) - (v[j] * z[l])
            A[l][j] = A[j][l]
        A[l][l] = A[l][l] - (2 * v[l] * z[l])
    
    A[n-1][n-1] = A[n-1][n-1] - (2 * v[n-1] * z[n-1])
    
    for j in range (k+2, n):
        A[k][j] = 0
        A[j][k] = 0

    A[k+1][k] = A[k+1][k] - (v[k+1] * z[k])
    A[k][k+1] = A[k+1][k]


# QR Algorithm


TOL = 1e-4              # tolerance
M = 100                 # maximum number of interations

k = 1
shift = 0

while k <= M:
    lamb = 0
    
    if numpy.absolute(A[n-2][n-1]) <= TOL:
        lamb = A[n-1][n-1] + shift
        print(lamb)
        n = n - 1
    if numpy.absolute(A[0][1]) <= TOL:
        lamb = A[0][0] + shift
        print(lamb)
        n = n - 1
        A[0][0] = A[1][1]
        for j in range (1, n):
            A[j][j] = A[j+1][j+1]
            A[j-1][j] = A[j][j+1]
            A[j][j-1] = A[j-1][j]

    if n == 0:
        break

    if n == 1:
        lamb = A[0][0] + shift
        print(lamb)
        break

    for j in range (2, n):
        if numpy.absolute(A[j-1][j]) <= TOL:
            print("split into") 
            break
    
    b = -(A[n-2][n-2] + A[n-1][n-1])
    c = A[n-1][n-1] * A[n-2][n-2] - (A[n-2][n-1]*A[n-2][n-1])   
    d = numpy.sqrt((b*b) - (4*c))

    micro1 = 0
    micro2 = 0
    if b > 0:
        micro1 = (-2*c)/(b+d)
        micro2 = -(b+d)/2
    else:
        micro1 = (d-b)/2
        micro2 = (2*c)/(d-b)
    
    lamb1 = 0
    lamb2 = 0
    if n == 2:
        lamb1 = micro1 + shift
        lamb2 = micro2 + shift
        print(lamb1)
        print(lamb2)
        break
    
    sigma = 0
    if numpy.absolute(micro1 - A[n-1][n-1]) < numpy.absolute(micro2 - A[n-1][n-1]):
        sigma = micro1
    else:
        sigma = micro2

    shift = shift + sigma
    

    dj = []
    xj = []
    yj = []
    zj = []
    qj = []
    rj = []
    cj = []
    rj = []
    sj = []


    for j in range (0, n):
        dj.append(A[j][j] - sigma)
        xj.append(0.0)
        yj.append(0.0)
        zj.append(0.0)
        qj.append(0.0)
        rj.append(0.0)
        cj.append(0.0)
        rj.append(0.0)
        sj.append(0.0)
    
    xj[0] = dj[0]
    yj[0] = A[0][1]

    for j in range (1, n):
        zj[j-1] = numpy.sqrt((xj[j-1] * xj[j-1]) + (A[j-1][j] * A[j-1][j]))
        
        cj[j] = xj[j-1]/zj[j-1]
        sj[j] = A[j-1][j]/zj[j-1]

        qj[j-1] = (cj[j] * yj[j-1]) + (sj[j] * dj[j])
        xj[j] = -(sj[j] * yj[j-1]) + (cj[j] * dj[j])

        if j != n-1:
            rj[j-1] = sj[j] * A[j][j+1]
            yj[j] = cj[j] * A[j][j+1]
    
    zj[n-1] = xj[n-1]
    A[0][0] = (sj[1] * qj[0]) + (cj[1] * zj[0])
    A[0][1] = sj[1]*zj[1]
    A[1][0] = A[0][1]

    for j in range (1, n-1):
        A[j][j] = (sj[j+1]*qj[j]) + (cj[j]*cj[j+1]*zj[j])
        A[j][j+1] = sj[j+1]*zj[j+1]
        A[j+1][j] = A[j][j+1]
    
    A[n-1][n-1] = cj[n-1]*zj[n-1]
    k = k+1
    
if k == M:
    print("Maximum number of iterations exceeded")