from scipy.special import gamma
from america_dataset import X_normalized as X
from america_dataset import y_normalized as y

year_1998 = [1767829, 275835018, 1550308707200, 710064652900, 944493975700, 4420214102700, 1]
year_1999 = [1753680, 279181581, 1680600618700, 739817241000, 1006948257700, 4840511600900, 1]
year_2000 = [1753680, 282398554, 1808549935800, 816712306100, 1071146818800, 5233099575200, 1]
def mul(X,Y):
    result = [[round(sum(a*b for a,b in zip(X_row,Y_col)),4) for Y_col in zip(*Y)] for X_row in X]
    return result

def subtract(X,Y):
    n = len(X)
    result = [[round(X[i][0] - Y[i][0],4)] for i in range(n)]
    return result

def power(X,a):
   n = len(X)
   result = [ [round(X[i][0] ** a,4)] for i in range(n) ]
   return result

def transpose(m):
    rez = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return rez

def loss_fct(X,params,y):
    h = mul(X,params)
    dif = subtract(h,y)
    sqr = [ (dif[i][0] ** 2) for i in range(len(dif))]
    sum = 0
    for i in range(len(sqr)):
       sum += sqr[i]
    return round(sum,4)


def integer_order_gd(X, y, learning_rate, num_iterations):
    m = len(X)
    n = len(X[0])
    params = [[0] for i in range(n)]
    for i in range(num_iterations):
       h = mul(X,params)
       dif = subtract(h,y)
       X_trans = transpose(X)
       gradient = mul(X_trans,dif)
       params = [ [round(params[j][0] - (1/m)*(gradient[j][0] * learning_rate),4)] for j in range(n) ]
    return params

def fractional_order_gd(X, y, learning_rate, alpha, num_iterations):
    m = len(X)
    n = len(X[0])
    params = [[0.1] for i in range(n)]
    for i in range(num_iterations):
       h = mul(X,params)
       dif = subtract(h,y)
       X_trans = transpose(X)
       gradient = mul(X_trans,dif)
       fractional_gradient = [[round(((params[j][0]**(1 - alpha))*gradient[j][0])/(gamma(1-alpha)*(1-alpha)*m),4)] for j in range(n)]
       params = [ [round(params[j][0] - (fractional_gradient[j][0] * learning_rate),4)] for j in range(n) ]
    return params


def MSE(predicted,y):
    m = len(y)
    sum = 0
    for i in range(m):
      sub_sum = round((predicted[i] - y[i][0])**2,4)
      sum += sub_sum
    return round(sum / m)
        
def ARE(predicted,real):
     result = abs(predicted - real) / (real)
     return result

def prediction(year,theta,year_num):
    min_minus = [-1775920, -217881437, -435096674400, -139246523300, -276260727000, -1217748969300,1]
    max_min =  [111630, 54514001, 997138271100, 571432092000, 633154685800, 2815223251600,1]
    min_y = 1752176414900
    max_y = 6391135825910
    year_normalized =  [round((year[i] + min_minus[i]) / (max_min[i]),4) for i in range(7)]
    year_normalized[6] = 1
    sum = 0
    for i in range(7):
       sum += (theta[i][0]*year_normalized[i])
    print("The value of GDP predicted for "+str(year_num)+" is:")
    print((sum*(max_y - min_y)) + min_y)

def result_of_integer_GD(X,y):
    theta = integer_order_gd(X,y,0.03,400)
    min_y = 1752176414900
    max_y = 6391135825910
    print("The parameters after fitting is:")
    print(theta)
    print("The value of loss function of integer order GD method is:")
    print(loss_fct(X,theta,y))
    mar = mul(X,theta)
    count = 1978
    print("The predicted results are:")
    for i in range(len(mar)):
      temp = (mar[i][0]*(max_y - min_y)) + min_y
      print(count,end="     ")
      print(temp)
      count += 1
    return theta

def result_of_fractional_GD(X,y):
    theta = fractional_order_gd(X,y,0.03,0.8,400)
    min_y = 1752176414900
    max_y = 6391135825910
    print("The parameters after fitting is:")
    print(theta)
    print("The value of loss function of fractional order GD method is:")
    print(loss_fct(X,theta,y))
    mar = mul(X,theta)
    count = 1978
    print("The predicted results are:")
    for i in range(len(mar)):
      temp = (mar[i][0]*(max_y - min_y)) + min_y
      print(count,end="     ")
      print(temp)
      count += 1
    return theta

# theta = result_of_integer_GD(X,y)
# theta = result_of_fractional_GD(X,y)
# prediction(year_1998,theta,1998)
# prediction(year_1999,theta,1999)
# prediction(year_2000,theta,2000)
