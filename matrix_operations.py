def dot_prdct(a, b):
    res = 0
    for i, j in zip(a, b):
        res += i * j
    return res


def matrixmul(inputs, layer):
    output = []
    for node in layer:
        result = []
        for example in inputs:
            if len(node) != len(example): raise Exception(
                "Nmbr input features != Nmbr of weights --> {} != {} ".format(len(example), len(node)))
            sum = dot_prdct(node, example)
            #sum += biases[layer.index(node)]
            result.append(round(sum, 3))
        output.append(result)
    return output


def transpose(array):
    nrow = len(array)
    ncol = len(array[0])
    new_row = []
    result = []
    for c in range(ncol):
        new_row = []
        for r in range(nrow):
            new_row.append(array[r][c])

        result.append(new_row)
    return result

if __name__ == '__main__':
    output = matrixmul(inputs, layer1)
    print((output))

    print(transpose(output))
