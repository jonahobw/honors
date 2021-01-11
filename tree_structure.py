from tree_helper import *
from tree_classes import *

def construct_trees(root, classifiers, signs, currentdepth, maxdepth):
    if(maxdepth<3):
        return "Error, max depth must be greater than 2"
    attribute = root.attribute_name
    values, split = split_signs(signs, attribute)

    #base case:
    #define the max depth as depth of the decision tree including the road sign layer and the final classifier layer
    #therefore once we hit max depth-2, it is time to stop and add the final classifiers/road signs, and return this
    #as the only tree possible
    if(currentdepth == maxdepth-2):
        limit_root = classifier(root.name, root.attribute_name)
        limit_tree = tree(limit_root)
        for i in range(len(split)):
            if(len(split[i])>1):
                limit_fin_classifier = final_classifier("Final classifier- " + str(attribute) + ": " +  str(values[i]), limit_root)
                limit_tree.add_node(limit_fin_classifier)
                limit_root.add_child(limit_fin_classifier)
                for Road_sign in split[i]:
                    limit_road_sign = road_sign(Road_sign.name, Road_sign.properties, limit_fin_classifier)
                    limit_fin_classifier.add_child(limit_road_sign)
                    limit_tree.add_node(limit_road_sign)
            else:
                limit_road_sign = road_sign(split[i][0].name, split[i][0].properties, limit_root)
                limit_root.add_child(limit_road_sign)
                limit_tree.add_node(limit_road_sign)
        return [limit_tree]

    #return array of possible trees with root as root
    tree_list = []

    possible_classifiers = find_child_classifiers(root, classifiers, signs)
    classifier_permutations = []
    get_permutations(possible_classifiers, classifier_permutations)

    for permutation in classifier_permutations:
        #all_trees:
        # 2d array where 1st dimension is each child, 2nd dimension are all the subtrees with that child as root
        all_trees = []
        for i in range(len(permutation)):
            if(permutation[i]==0):
                only_sign = split[i][0]
                new_sign = road_sign(only_sign.name, only_sign.properties)
                single_sign_tree = tree(new_sign)
                all_trees.append([single_sign_tree])
            elif(permutation[i] == 1):
                #make final classifier and add all road signs to it, then make a tree and add to all_signs
                final_signs = split[i]
                fin_classifier = final_classifier("Final classifier- " + str(attribute) + ": " +  str(values[i]))
                fin_classifier_tree = tree(fin_classifier)
                for sign in final_signs:
                    new_sign = road_sign(sign.name, sign.properties, fin_classifier)
                    fin_classifier.add_child(new_sign)
                    fin_classifier_tree.add_node(new_sign)
                all_trees.append([fin_classifier_tree])
            else:
                child_node = classifier(values[i], permutation[i])
                #trees: list of possible subtrees with this child as root
                new_classifiers_list = classifiers.copy()
                new_classifiers_list.remove(attribute)
                trees = construct_trees(child_node, new_classifiers_list, split[i], currentdepth+1, maxdepth)
                all_trees.append(trees)
        #child_tree_permutations = list of possible permutations of child trees
        child_tree_permutations = []
        get_permutations(all_trees, child_tree_permutations)
        for child_tree_permutation in child_tree_permutations:
            # create a copy of current root node, copy of each child tree, and put them together in a new tree
            # and add this to tree list
            root_copy = classifier(root.name, root.attribute_name)
            new_tree = tree(root_copy)
            for subtree in child_tree_permutation:
                #create a copy of the subtree
                subtree_copy = copy_tree(subtree)
                #add the root of the subtree to be a child of the overall root
                root_copy.add_child(subtree_copy.root)
                #add the overall root as the parent of the subtree root
                subtree_copy.root.parent = root_copy
                #add subtree nodes to the new_tree nodes list
                new_tree.nodes.extend(subtree_copy.nodes)
            tree_list.append(new_tree)
    return tree_list


def find_optimal_tree(maxdepth, weighted = False):
    # weighted indicates whether or not to used weighted average attack distance
    signsarray = signs()
    classifiers = list(signsarray[0].properties.keys())
    max_avg_attack_distance = 0
    optimal_tree = None
    for Classifier in classifiers:
        root = classifier("root", Classifier)
        treelist = construct_trees(root, classifiers, signsarray, 1, maxdepth)
        print(str(len(treelist)) + " Trees found for root node of", Classifier)
        local_max_avg_attack_distance = 0
        for Tree in treelist:
            if weighted:
                aad = weighted_avg_attack_distance(Tree, signsarray)
            else:
                aad = avg_attack_distance(Tree, signsarray)
            if(aad> local_max_avg_attack_distance):
                local_max_avg_attack_distance = aad
            if(aad>max_avg_attack_distance):
                optimal_tree = Tree
                max_avg_attack_distance = aad
        print("Max AAD found was ", str(local_max_avg_attack_distance))

    return optimal_tree


def construct_trees_classifiers(root, classifiers, signs, current_count, max_count):
    if (max_count < 3):
        return "Error, max depth must be greater than 2"
    attribute = root.attribute_name
    values, split = split_signs(signs, attribute)

    # base case:
    # define the max count as the number of classifiers including final classifiers
    if (current_count == max_count - 2):
        limit_root = classifier(root.name, root.attribute_name)
        limit_tree = tree(limit_root)
        for i in range(len(split)):
            if (len(split[i]) > 1):
                limit_fin_classifier = final_classifier("Final classifier- " + str(attribute) + ": " + str(values[i]),
                                                        limit_root)
                limit_tree.add_node(limit_fin_classifier)
                limit_root.add_child(limit_fin_classifier)
                for Road_sign in split[i]:
                    limit_road_sign = road_sign(Road_sign.name, Road_sign.properties, limit_fin_classifier)
                    limit_fin_classifier.add_child(limit_road_sign)
                    limit_tree.add_node(limit_road_sign)
            else:
                limit_road_sign = road_sign(split[i][0].name, split[i][0].properties, limit_root)
                limit_root.add_child(limit_road_sign)
                limit_tree.add_node(limit_road_sign)
        return [limit_tree]

    # return array of possible trees with root as root
    tree_list = []

    possible_classifiers = find_child_classifiers(root, classifiers, signs)
    classifier_permutations = []
    get_permutations(possible_classifiers, classifier_permutations)

    for permutation in classifier_permutations:
        # all_trees:
        # 2d array where 1st dimension is each child, 2nd dimension are all the subtrees with that child as root
        all_trees = []
        for i in range(len(permutation)):
            if (permutation[i] == 0):
                only_sign = split[i][0]
                new_sign = road_sign(only_sign.name, only_sign.properties)
                single_sign_tree = tree(new_sign)
                all_trees.append([single_sign_tree])
            elif (permutation[i] == 1):
                # make final classifier and add all road signs to it, then make a tree and add to all_signs
                final_signs = split[i]
                fin_classifier = final_classifier("Final classifier- " + str(attribute) + ": " + str(values[i]))
                fin_classifier_tree = tree(fin_classifier)
                for sign in final_signs:
                    new_sign = road_sign(sign.name, sign.properties, fin_classifier)
                    fin_classifier.add_child(new_sign)
                    fin_classifier_tree.add_node(new_sign)
                all_trees.append([fin_classifier_tree])
            else:
                child_node = classifier(values[i], permutation[i])
                # trees: list of possible subtrees with this child as root
                new_classifiers_list = classifiers.copy()
                new_classifiers_list.remove(attribute)
                trees = construct_trees(child_node, new_classifiers_list, split[i], current_count + 1, max_count)
                all_trees.append(trees)
        # child_tree_permutations = list of possible permutations of child trees
        child_tree_permutations = []
        get_permutations(all_trees, child_tree_permutations)
        for child_tree_permutation in child_tree_permutations:
            # create a copy of current root node, copy of each child tree, and put them together in a new tree
            # and add this to tree list
            root_copy = classifier(root.name, root.attribute_name)
            new_tree = tree(root_copy)
            for subtree in child_tree_permutation:
                # create a copy of the subtree
                subtree_copy = copy_tree(subtree)
                # add the root of the subtree to be a child of the overall root
                root_copy.add_child(subtree_copy.root)
                # add the overall root as the parent of the subtree root
                subtree_copy.root.parent = root_copy
                # add subtree nodes to the new_tree nodes list
                new_tree.nodes.extend(subtree_copy.nodes)
            tree_list.append(new_tree)
    return tree_list


def construct_tree(signsarray):
    root = classifier("root", "shape")
    Tree = tree(root)
    values, split = split_signs(signsarray, "shape")
    for value in values:
        i = values.index(value)
        signs = split[i]
        if(len(signs)<2):
            root.add_child(signs[0])
            signs[0].parent = root
            Tree.add_node(signs[0])
            continue

        values1, split1 = split_signs(signs, "color")
        new_classifier = classifier(value, "color", root)
        root.add_child(new_classifier)
        Tree.add_node(new_classifier)


        for val in values1:
            nc = final_classifier(str(value + " " + val))
            nc.parent = new_classifier
            new_classifier.add_child(nc)
            Tree.add_node(nc)
            j = values1.index(val)
            fin_signs = split1[j]
            for f in fin_signs:
                nc.add_child(f)
                Tree.add_node(f)
                f.parent = nc
    return Tree


Tree = find_optimal_tree(4, False)
Tree_stats(Tree, False)
