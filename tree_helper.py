from tree_classes import road_sign, classifier, final_classifier, tree

def signs():
    signs = []
    for i in range(43):
        signs.append([])
    signs[0] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": True, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[1] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": True, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[2] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": True, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[3] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": True, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[4] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": True, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[5] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": True, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[6] = {"color": "white", "fill": "white", "shape": "circle", "car": False, "numbers": True, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[7] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": True, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[8] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": True, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[9] = {"color": "red", "fill": "white", "shape": "circle", "car": True, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[10] = {"color": "red", "fill": "white", "shape": "circle", "car": True, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[11] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": True}
    signs[12] = {"color": "yellow", "fill": "yellow", "shape": "diamond", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[13] = {"color": "red", "fill": "white", "shape": "inverted_triangle", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[14] = {"color": "red", "fill": "red", "shape": "octagon", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[15] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[16] = {"color": "red", "fill": "white", "shape": "circle", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[17] = {"color": "red", "fill": "red", "shape": "circle", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[18] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": False}
    signs[19] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": True}
    signs[20] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                "curved_arrows": False, "road": True}
    signs[21] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": True}
    signs[22] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[23] = {"color": "red", "fill": "white", "shape": "triangle", "car": True, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[24] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": True}
    signs[25] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[26] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[27] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[28] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[29] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[30] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[31] = {"color": "red", "fill": "white", "shape": "triangle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[32] = {"color": "white", "fill": "white", "shape": "circle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[33] = {"color": "blue", "fill": "blue", "shape": "circle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": True, "road": False}
    signs[34] = {"color": "blue", "fill": "blue", "shape": "circle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": True, "road": False}
    signs[35] = {"color": "blue", "fill": "blue", "shape": "circle", "car": False, "numbers": False, "straight_arrows": True,
                 "curved_arrows": False, "road": False}
    signs[36] = {"color": "blue", "fill": "blue", "shape": "circle", "car": False, "numbers": False, "straight_arrows": True,
                 "curved_arrows": True, "road": False}
    signs[37] = {"color": "blue", "fill": "blue", "shape": "circle", "car": False, "numbers": False, "straight_arrows": True,
                 "curved_arrows": True, "road": False}
    signs[38] = {"color": "blue", "fill": "blue", "shape": "circle", "car": False, "numbers": False, "straight_arrows": True,
                 "curved_arrows": False, "road": False}
    signs[39] = {"color": "blue", "fill": "blue", "shape": "circle", "car": False, "numbers": False, "straight_arrows": True,
                 "curved_arrows": False, "road": False}
    signs[40] = {"color": "blue", "fill": "blue", "shape": "circle", "car": False, "numbers": False, "straight_arrows": False,
                 "curved_arrows": True, "road": False}
    signs[41] = {"color": "white", "fill": "white", "shape": "circle", "car": True, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}
    signs[42] = {"color": "white", "fill": "white", "shape": "circle", "car": True, "numbers": False, "straight_arrows": False,
                 "curved_arrows": False, "road": False}

    road_sign_objects = []
    i = 0
    for sign in signs:
        road_sign_objects.append(road_sign(str(i), sign))
        i+=1
    return road_sign_objects


def attack_danger_weights(startsign_name, targetsign_name):

    '''

    for future use: the code to print out the weights:
    a = []
    for i in range(43):
        array = []
        for j in range(43):
            if(i==j):
                array.append(0)
            else:
                array.append(j)
        a.append(array)

    for arr in a:
        print("\n# start class " + str(i))
        print(str(arr) + ",")
    '''

    weights = [
        # start class 0
        [0, 3, 4, 4, 5, 6, 2, 7, 8, 2, 2, 2, 2, 2, 3, 5, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2],

        # start class 1
        [1, 0, 3, 4, 4, 5, 2, 7, 8, 2, 2, 2, 2, 2, 4, 5, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2],

        # start class 2
        [4, 3, 0, 2, 3, 4, 3, 5, 6, 3, 3, 3, 3, 3, 5, 6, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3],

        # start class 3
        [4, 3, 2, 0, 2, 3, 3, 4, 6, 3, 3, 3, 3, 4, 5, 7, 3, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3],

        # start class 4
        [5, 4, 2, 2, 0, 2, 4, 4, 5, 4, 4, 4, 4, 4, 6, 7, 4, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
         4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4],

        # start class 5
        [6, 5, 4, 3, 2, 0, 4, 3, 4, 4, 4, 4, 4, 4, 8, 8, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
         4, 4, 4, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4],

        # start class 6
        [2, 2, 3, 4, 5, 6, 0, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 7
        [7, 6, 5, 4, 4, 3, 5, 0, 3, 5, 5, 5, 5, 5, 8, 9, 5, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         5, 5, 5, 6, 6, 6, 6, 6, 5, 5, 4, 5, 5],

        # start class 8
        [8, 7, 6, 5, 4, 3, 6, 2, 0, 6, 6, 6, 6, 6, 9, 10, 6, 10, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
         6, 6, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6],

        # start class 9
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 0, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 10
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 0, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 11
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 0, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 12
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 0, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 13
        [2, 3, 4, 4, 5, 6, 4, 7, 8, 4, 4, 4, 6, 0, 5, 6, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4,
         4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],

        # start class 14
        [6, 6, 6, 7, 8, 8, 5, 9, 10, 6, 6, 6, 6, 4, 0, 4, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],

        # start class 15
        [5, 6, 6, 7, 8, 8, 5, 9, 10, 5, 5, 5, 6, 4, 3, 0, 5, 1, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],

        # start class 16
        [2, 2, 3, 4, 5, 6, 2, 7, 8, 2, 2, 2, 2, 3, 5, 3, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],

        # start class 17
        [5, 6, 6, 7, 8, 8, 5, 9, 10, 5, 5, 5, 6, 4, 3, 1, 5, 0, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],

        # start class 18
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 19
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 20
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 21
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 22
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 23
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 24
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 25
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 26
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 27
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 28
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 29
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 30
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 31
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 0, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 32
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 0, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],

        # start class 33
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 0, 4, 4, 3, 4, 1, 1, 3, 1, 1],

        # start class 34
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 4, 0, 4, 4, 3, 1, 1, 3, 1, 1],

        # start class 35
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 4, 4, 0, 3, 3, 1, 1, 3, 1, 1],

        # start class 36
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 3, 4, 3, 0, 3, 1, 1, 3, 1, 1],

        # start class 37
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 4, 3, 3, 3, 0, 1, 1, 3, 1, 1],

        # start class 38
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 2, 1, 1],

        # start class 39
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 0, 2, 1, 1],

        # start class 40
        [2, 3, 4, 4, 5, 6, 3, 7, 8, 3, 3, 3, 3, 2, 2, 4, 4, 4, 2, 2, 2, 1, 2, 2, 3, 3, 2, 2, 2, 3,
         3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 0, 3, 3],

        # start class 41
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 0, 1],

        # start class 42
        [2, 2, 3, 4, 5, 6, 1, 7, 8, 1, 1, 1, 1, 3, 5, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 0],
    ]

    return weights[int(startsign_name)][int(targetsign_name)]


def split_signs(sign_array, attribute_name, dict = False):

    # splits the signs in signarray based on the attribute_name
    # return values:
    # values (array) - unique values of attribute name
    # split (2d array) - the split up signs in sets, with the indexes
    #                   corresponding to the values array
    # dict (bool)    - whether or not to return dictionary format

    split = []
    values = []
    for sign in sign_array:
        if(sign.properties[attribute_name] not in values):
            values.append(sign.properties[attribute_name])
            split.append([sign])
        else:
            a = values.index(sign.properties[attribute_name])
            split[a].append(sign)
    if dict:
        split_dict = {}
        for i, value in enumerate(values):
            split_dict[value] = split[i]
        return split_dict
    return values, split


def split_all(attribute):
    # splits all 43 signs based on the attribute, returns an array of length 43 where each index corresponds to
    # a class of a road sign and the element at that index represents the value of that class when split by the
    # attribute
    signsarray = signs()
    values = []
    for i in range(len(signsarray)):
        for sign in signsarray:
            if(int(sign.name) == i):
                values.append(sign.properties[attribute])
                break
    return values


def find_child_classifiers(root, classifiers, signs):
    attribute = root.attribute_name
    values, split = split_signs(signs, attribute)
    possible_classifiers = []
    for i in range(len(split)):
        if (len(split[i]) == 1):
            # there is only 1 sign in this group
            # [0] indicates that there is only 1 sign
            # this sign will need to be a direct child of this root node
            possible_classifiers.append([0])
        elif(len(split[i]) == 2):
            #there are only 2 signs in this group
            #even if there is a classifier that distinguishes them, the next level down will be the road signs
            #themselves, so it makes sense to have a final classifier here
            possible_classifiers.append([1])
        else:
            # if the length of split[i] is more than 2, figure out which of the remaining
            # classifiers will split the group into 2 or more groups
            current_set_possible_classifiers = []
            for classifier in classifiers:
                child_values, child_split = split_signs(split[i], classifier)
                if (len(child_split) > 1):
                    current_set_possible_classifiers.append(classifier)
            if (len(current_set_possible_classifiers) == 0):
                # there are no classifiers that split the group of signs into 2 or more groups
                # [1] indicates that there are multiple signs, but no classifiers to differentiate between them
                # a final classifier will be needed
                possible_classifiers.append([1])
            else:
                possible_classifiers.append(current_set_possible_classifiers)
    return possible_classifiers


def find_child_classifiers2(root, classifiers, signs, classifier_budget):
    attribute = root.attribute_name
    values, split = split_signs(signs, attribute)
    possible_classifiers = []
    for i in range(len(split)):
        if (len(split[i]) == 1):
            # there is only 1 sign in this group
            # [0] indicates that there is only 1 sign
            # this sign will need to be a direct child of this root node
            possible_classifiers.append([0])
        elif(len(split[i]) == 2):
            #there are only 2 signs in this group
            #even if there is a classifier that distinguishes them, the next level down will be the road signs
            #themselves, so it makes sense to have a final classifier here
            possible_classifiers.append([1])
        else:
            # if the length of split[i] is more than 2, figure out which of the remaining
            # classifiers will split the group into 2 or more groups
            current_set_possible_classifiers = []
            for classifier in classifiers:
                child_values, child_split = split_signs(split[i], classifier)
                if (len(child_split) > 1):
                    current_set_possible_classifiers.append(classifier)
            if (len(current_set_possible_classifiers) == 0):
                # there are no classifiers that split the group of signs into 2 or more groups
                # [1] indicates that there are multiple signs, but no classifiers to differentiate between them
                # a final classifier will be needed
                possible_classifiers.append([1])
            else:
                possible_classifiers.append(current_set_possible_classifiers)
    return possible_classifiers


def print_tree(tree):

    # returns a 2D array where the 1st dimension is the depth of the nodes
    # and the second dimension are the nodes at that depth
    # nodes are printed in tuples (name of the node, name of the parent node)

    nodes = []
    nodenames = []
    root = tree.root
    nodenames.append([(type(root), root.name, root.attribute_name, root.parent)])
    nodes.append([root])

    children = root.children
    while(len(children)>0):
        grandchildren = []
        nodes.append(children)
        add_names = []
        for node in children:
            if(isinstance(node, classifier)):
                add_names.append((type(node), node.name, node.attribute_name, node.parent.name))
            else:
                add_names.append((type(node), node.name, node.parent.name))
            if(len(node.children)>1):
                for grandchild in node.children:
                    grandchildren.append(grandchild)
        nodenames.append(add_names)
        children = grandchildren
    print(nodenames)


def count_road_signs(Tree):
    # counts the number of road sign objects in a tree
    root = Tree.root
    unexplored = [root]
    count = 0
    while(len(unexplored)>0):
        node = unexplored.pop()
        if(isinstance(node, road_sign)):
            count +=1
        unexplored += node.children
    return count


def count_leaf_nodes(Tree):
    # counts the number of road sign objects in a tree
    root = Tree.root
    unexplored = [root]
    count = 0
    while(len(unexplored)>0):
        node = unexplored.pop()
        if(len(node.children)<1):
            count +=1
            continue
        unexplored += node.children
    return count


def count_all_classifiers(Tree):
    return len(Tree.nodes)- count_leaf_nodes(Tree)


def copy_tree(Tree):
    root = Tree.root
    if(isinstance(root, road_sign)):
        copy_root = road_sign(root.name, root.properties)
    elif(isinstance(root, classifier)):
        copy_root = classifier(root.name, root.attribute_name)
    elif(isinstance(root, final_classifier)):
        copy_root = final_classifier(root.name)
    copied_tree = tree(copy_root)
    copy_children(copy_root, root.children, copied_tree)
    return copied_tree


def copy_children(copy_parent, real_children, copy_tree):
    if(len(real_children)<1):
        return
    else:
        for real_child in real_children:
            if (isinstance(real_child, classifier)):
                newclassifier = classifier(real_child.name, real_child.attribute_name, copy_parent)
                copy_tree.add_node(newclassifier)
                copy_parent.children.append(newclassifier)
                copy_children(newclassifier, real_child.children, copy_tree)
            if (isinstance(real_child, final_classifier)):
                newfinalclassifier = final_classifier(real_child.name, copy_parent)
                copy_tree.add_node(newfinalclassifier)
                copy_parent.children.append(newfinalclassifier)
                copy_children(newfinalclassifier, real_child.children, copy_tree)
            if (isinstance(real_child, road_sign)):
                newroadsign = road_sign(real_child.name, real_child.properties, copy_parent)
                copy_tree.add_node(newroadsign)
                copy_parent.children.append(newroadsign)


def find_road_sign_path(root, sign):
    path = []
    current = root
    road_sign_numbers = []
    target = sign.name
    for child in current.children:
        if(isinstance(child, road_sign)):
            road_sign_numbers.append(child.name)
    bool = target in road_sign_numbers
    while(not bool):
        path.append(current)
        # figure out which child node to go to

        # if current node is a classifier, figure out which child
        # is the right attribute
        if(isinstance(current, classifier)):
            attribute = sign.properties[current.attribute_name]
            for node in current.children:
                if(node.name == attribute):
                    current = node
                    break
                if(isinstance(node, final_classifier)):
                    if(node.name.find(str(attribute))>0):
                        current = node
                        break
        road_sign_numbers = []
        for child in current.children:
            if (isinstance(child, road_sign)):
                road_sign_numbers.append(child.name)
        bool = target in road_sign_numbers
    path.append(current)
    path.append(sign)
    return path


def attack_distance(root, startsign, endsign):
    path1 = find_road_sign_path(root, startsign)
    path2 = find_road_sign_path(root, endsign)
    while(path1[0] == path2[0]):
        path1 = path1[1:]
        path2 = path2[1:]
    return len(path2)


def avg_attack_distance(tree, road_signs):
    # it is a directed attack so we want the data going from 0 -> 1 and 1 -> 0
    root = tree.root
    total_distance = 0
    count = 0
    signs = road_signs.copy()
    for startnode in signs:
        for endnode in signs:
            if (startnode.name == endnode.name):
                continue
            a = attack_distance(root, startnode, endnode)
            total_distance += a
            #print("", startnode.name, endnode.name, a)
            count +=1
    if(count != 1806):
        a = "Error, count = " + str(count)
        return a
    return total_distance/count


def weighted_avg_attack_distance(tree, road_signs):
    # it is a directed attack so we want the data going from 0 -> 1 and 1 -> 0
    root = tree.root
    total_distance = 0
    count = 0
    signs = road_signs.copy()
    for startnode in signs:
        for endnode in signs:
            if (startnode.name == endnode.name):
                continue
            a = attack_distance(root, startnode, endnode)
            total_distance += a*attack_danger_weights(startnode.name, endnode.name)
            #print("", startnode.name, endnode.name, a)
            count +=1
    if(count != 1806):
        a = "Error, count = " + str(count)
        return a
    return total_distance/count


def get_permutations(array, permutations = [], current = []):
    #takes a 2d array and finds all permutations of it, taking one element from each array

    #base case
    if len(array) == 1:
        if(array[0] == []):
            new = current.copy()
            new.append([])
            permutations.append(new)
            return
        for item in array[0]:
            new = current.copy()
            new.append(item)
            permutations.append(new)
        return
    else:
    #len(2d_array is greater than 1)
        if(array[0] == []):
            new = current.copy()
            new.append([])
            get_permutations(array[1:], permutations, new)
        else:
            for item in array[0]:
                new = current.copy()
                new.append(item)
                get_permutations(array[1:], permutations, new)


def Tree_stats(Tree, weighted = False):
    signsarray = signs()
    print("\n\n\nThe tree is:")
    print(print_tree(Tree))
    if weighted:
        print("\nweighted average attack distance:")
        print(weighted_avg_attack_distance(Tree, signsarray))
    else:
        print("\naverage attack distance:")
        print(avg_attack_distance(Tree, signsarray))
    print("number of road signs:")
    print(count_road_signs(Tree))
    print("number of leaf nodes:")
    print(count_leaf_nodes(Tree))
    print("total number of nodes")
    print(len(Tree.nodes))
    print("total number of classifiers")
    print(count_all_classifiers(Tree))
