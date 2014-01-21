k = 3

for r in range(-1, k):
    for j in range(k):
        print "==="
        print "r = %d, j = %d" % (r, j)
        for m in range(j+1, k+1):
            # print "m = %d" % m
            sumOfProducts = 0
            prod_denom = 1
            for l in range(0, k+1):
                if l == m:
                    continue

                # print "l = %d" % l

                prod = 1
                for q in range(0, k+1):
                    if q == l or q == m:
                        continue

                    # print "q = %d" % q

                    prod *= r - q + 1

                sumOfProducts += prod
                prod_denom *= m - l

            print "%f / %f" % (sumOfProducts, prod_denom)
