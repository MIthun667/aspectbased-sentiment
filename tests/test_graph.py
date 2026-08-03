from src.data.graph import (
    build_dependency_edges,
    convert_heads_to_zero_based,
    shortest_distances_from_aspect,
)


def test_convert_corenlp_heads() -> None:
    raw_heads = [2, 0, 2]

    converted, roots = convert_heads_to_zero_based(
        raw_heads,
        number_of_tokens=3,
    )

    assert converted == [1, -1, 1]
    assert roots == [1]


def test_dependency_edges() -> None:
    heads = [1, -1, 1]
    relations = ["nsubj", "root", "dobj"]

    edges = build_dependency_edges(
        heads,
        relations,
        add_reverse_edges=True,
        add_self_loops=True,
    )

    edge_lists = [edge.to_list() for edge in edges]

    assert [1, 0, "nsubj", "forward"] in edge_lists
    assert [0, 1, "nsubj__reverse", "reverse"] in edge_lists
    assert [1, 2, "dobj", "forward"] in edge_lists
    assert [2, 1, "dobj__reverse", "reverse"] in edge_lists
    assert [0, 0, "self_loop", "self"] in edge_lists


def test_shortest_aspect_distances() -> None:
    # Tree: 0 -- 1 -- 2 -- 3
    heads = [-1, 0, 1, 2]

    distances = shortest_distances_from_aspect(
        dependency_heads=heads,
        aspect_indices=[2],
    )

    assert distances == [2, 1, 0, 1]


def test_all_pairs_shortest_distances() -> None:
    from src.data.graph import all_pairs_shortest_distances

    # Tree:
    #
    # 0 -- 1 -- 2
    #      |
    #      3
    dependency_heads = [1, -1, 1, 1]

    observed = all_pairs_shortest_distances(
        dependency_heads
    )

    expected = [
        [0, 1, 2, 2],
        [1, 0, 1, 1],
        [2, 1, 0, 2],
        [2, 1, 2, 0],
    ]

    assert observed == expected


def test_all_pairs_shortest_distances_clipping() -> None:
    from src.data.graph import all_pairs_shortest_distances

    # Chain: 0 -- 1 -- 2 -- 3 -- 4 -- 5
    dependency_heads = [-1, 0, 1, 2, 3, 4]

    observed = all_pairs_shortest_distances(
        dependency_heads,
        maximum_distance=3,
    )

    assert observed[0] == [0, 1, 2, 3, 3, 3]
    assert observed[5] == [3, 3, 3, 2, 1, 0]


def test_all_pairs_matrix_is_symmetric() -> None:
    from src.data.graph import all_pairs_shortest_distances

    dependency_heads = [1, -1, 1, 2, 2]

    matrix = all_pairs_shortest_distances(
        dependency_heads
    )

    for row_index, row in enumerate(matrix):
        assert row[row_index] == 0

        for column_index in range(len(matrix)):
            assert (
                matrix[row_index][column_index]
                == matrix[column_index][row_index]
            )
