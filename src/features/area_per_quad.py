from feature_difference import scalar_feature_diff


def get_area_by_quadrant(quadrants):
    areas = []
    for quad_num in range(4):
        areas.append(len([pixel for row in quadrants for pixel in row if pixel == quad_num]))
    return areas


def f_area_quad(quadrants):
    areas = get_area_by_quadrant(quadrants)
    return scalar_feature_diff(areas)
