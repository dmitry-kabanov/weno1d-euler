k = 3
prod_list = []
prod_denom_list = []

for j in range(2 * k - 1):
    print "===="
    print "mu %s" % j

    for m in range(j + 1, 2 * k):
        numerator = 0.0
        prod_denom = 1
        for l in range(0, 2 * k):
            if l == m:
                continue

            prod = 1
            for q in range(0, 2 * k):
                if q == m or q == l:
                    continue

                prod *= k - q

            numerator += prod
            prod_denom *= m - l
            # if prod != 0:
            #     prod_list.append(prod)
            #     prod_denom_list.append(prod_denom)

        print "%d / %d" % (numerator, prod_denom)
    # sum = 0.0
    # common_denom = 1
    # for i in range(len(prod_denom_list)):
    #     common_denom *= prod_denom_list[i]
    #
    # for i in range(len(prod_list)):
    #     prod_list[i] *= common_denom / float(prod_denom_list[i])
    #     sum += prod_list[i]
    #
    # print "%f / %f" % (sum, common_denom)
    # print sum / float(common_denom)

    # for i in range(len(prod_list)):
    #     print "%d / %d" % (prod_list[i], prod_denom_list[i])
