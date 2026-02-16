def ticket_price(ticket,student,height):
    if (student == 1) or (120 < height <= 150):
        return ticket/2
    if height <=120:
        return 0
    return ticket

y1=ticket_price(120, 0, 120)
y2=ticket_price(120, 1, 120)
y3=ticket_price(120, 0, 130)
y4=ticket_price(120, 1, 130)
y5=ticket_price(120, 0, 175)
y6=ticket_price(120, 1, 175)

print(y1)
print(y2)
print(y3)
print(y4)
print(y5)
print(y6)



def pi_4(n):
    result = 0
    for i in range(1,n+1):
        result = result + (-1)**n*(1/(2*n-1))
    return result

print(pi_4(10))
print(pi_4(100))
print(pi_4(150))
print(pi_4(200))