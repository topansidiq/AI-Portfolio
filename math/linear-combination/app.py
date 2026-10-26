from helper import Finance

cash = Finance()

print(cash.total([100, 200, 300]))
print(cash.average([100, 200, 300], [90, 80], 90, 230, [1, 3, 1000]))
