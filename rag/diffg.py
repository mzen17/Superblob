from rag.graph import loadgraph, entity_collapse, query

def graphdiff(graphA, graphB, uniongraph, simtr, uniontr):
    """Compare two graphs and find differences in edge labels.
    
    1. Creates union of both graphs
    2. Finds unique edges using entity_collapse (only caring about edge labels)
    3. Generates RAGs from each graph using entity_collapse
    4. For each unique edge, queries both RAGs and builds comparison map
    
    Returns: list of dicts [{edge: str, queryA_set: [images], queryB_set: [images]}]
    """
    
    uniq_edge_results = entity_collapse(uniongraph, uniontr)
    unique_edges = [edge_label for edge_label, images in uniq_edge_results]
        
    # Build comparison map
    data = []
    for edge in unique_edges:
        resultA = query(graphA, edge, return_similarity=True)
        resultB = query(graphB, edge, return_similarity=True)
        
        if not resultA or not resultB:
            continue
        
        labelA, queryA_set, simA = resultA
        labelB, queryB_set, simB = resultB
        
        # Skip if the matched label is different from the query edge
        # or if similarity is too low (< 0.2)
        if simA < simtr or simB < simtr:
            continue
        
        data.append({
            'edge': edge,
            'queryA_set': queryA_set,
            'queryB_set': queryB_set
        })
    
    return data

def jaccard(diffdata):
    """Calculate Jaccard index for each edge in the diff data.    
    Args:
        diffdata: list of dicts with structure [{edge: str, queryA_set: [images], queryB_set: [images]}]
    
    Returns:
        tuple: (jaccard_scores dict, overall_jaccard float)
            - jaccard_scores: dict mapping edge labels to their Jaccard indices
            - overall_jaccard: Jaccard index of merged queryA_set vs merged queryB_set
    """
    jaccard_scores = {}
    
    # Accumulate all images from both sets
    all_queryA = set()
    all_queryB = set()
    
    for item in diffdata:
        setA = set(item['queryA_set'])
        setB = set(item['queryB_set'])
        
        # Accumulate for overall calculation
        all_queryA.update(setA)
        all_queryB.update(setB)
        
        # Calculate intersection and union
        intersection = setA & setB
        union = setA | setB
        
        # Calculate Jaccard index (handle empty union case)
        if len(union) == 0:
            jaccard_index = 0.0
        else:
            jaccard_index = len(intersection) / len(union)
        
        jaccard_scores[item["edge"]] = jaccard_index
    
    # Calculate mean Jaccard index from all edge scores
    if len(jaccard_scores) == 0:
        mean_jaccard = 0.0
    else:
        mean_jaccard = sum(jaccard_scores.values()) / len(jaccard_scores)
    
    # Sort by Jaccard index (descending order)
    jaccard_scores = dict(sorted(jaccard_scores.items(), key=lambda x: x[1], reverse=True))
    
    print(jaccard_scores)
    print(f"Mean Jaccard: {mean_jaccard}")
    return jaccard_scores, mean_jaccard


def matrixdiff(graph_set, tr=0.6, simtr=0.30, uniontr=0.4):
    n = len(graph_set)

    tcgraphs = [loadgraph(graph) for graph in graph_set]
    tcrags = [entity_collapse(g, tr) for g in tcgraphs]

    ans = [[-1.0 for _ in range(n)] for _ in range(n)]
    debug = [[{} for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            print(f"Working on {i} {j}")
            if i == j:
                ans[i][j] = 1
        
            elif ans[j][i] != -1:
                ans[i][j] = ans[j][i]
            
            else:
                diffcompute = graphdiff(tcrags[i], tcrags[j], tcgraphs[i] + tcgraphs[j], simtr, uniontr)
                debug[i][j], ans[i][j] = jaccard(diffcompute)

    return debug, ans