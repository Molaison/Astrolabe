import ast

import numpy as np

from astrolabe.selectors.registry import SelectorRegistry


class _SelectorExpressionTransformer(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            op_cls = ast.BitAnd
        elif isinstance(node.op, ast.Or):
            op_cls = ast.BitOr
        else:
            return node

        expr = values[0]
        for value in values[1:]:
            expr = ast.copy_location(
                ast.BinOp(left=expr, op=op_cls(), right=value), node
            )
        return expr

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(ast.UnaryOp(op=ast.Invert(), operand=operand), node)
        return ast.copy_location(ast.UnaryOp(op=node.op, operand=operand), node)


def construct_position_selector(
    expression, evaluation_result, protomers
):
    default_sites = {name: np.ones(protomer.length, dtype=bool) for name, protomer in protomers.items()}
    if expression == "None" or expression is None:
        return default_sites

    position_vars = extract_variables(expression)
    complex_poses = getattr(evaluation_result, "complex_poses", None)
    if complex_poses is None:
        pose_vars = position_vars & SelectorRegistry.pose_required_variables()
        if pose_vars:
            complex_paths = getattr(evaluation_result, "complex_paths", None)
            if not complex_paths:
                raise ValueError(
                    "Position selector expression requires evaluated complex paths "
                    "to construct ComplexPoses."
                )
            from astrolabe.evaluators.utils import ComplexPoses

            path_only_vars = SelectorRegistry.path_only_variables()
            if pose_vars <= path_only_vars:
                complex_poses = ComplexPoses({})
                complex_poses.complex_paths = dict(complex_paths)
            else:
                complex_poses = ComplexPoses.construct_complex_poses(evaluation_result)
                complex_poses.complex_paths = dict(complex_paths)
            try:
                evaluation_result.complex_poses = complex_poses
            except Exception:
                pass
        else:
            from astrolabe.evaluators.utils import ComplexPoses

            complex_poses = ComplexPoses({})
    else:
        complex_paths = getattr(evaluation_result, "complex_paths", None)
        if complex_paths and not hasattr(complex_poses, "complex_paths"):
            complex_poses.complex_paths = dict(complex_paths)

    env = SelectorRegistry.construct_environment(
        position_vars,
        complex_poses=complex_poses,
        evaluation_result=evaluation_result,
        protomers=protomers,
    )

    tree = ast.parse(expression, mode="eval")
    tree = _SelectorExpressionTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    eval_globals = {"__builtins__": {}}
    eval_globals.update(env)
    final_selector = eval(compile(tree, "<position_selector>", "eval"), eval_globals)


    if final_selector is None:
        return default_sites

    if isinstance(final_selector, bool):
        if final_selector:
            return default_sites
        return {
            proto: np.zeros_like(mask, dtype=bool)
            for proto, mask in default_sites.items()
        }

    result = {}
    for proto, base_mask in default_sites.items():
        selector_mask = final_selector.sites.get(proto)
        if selector_mask is None:
            result[proto] = base_mask.copy()
        else:
            result[proto] = np.logical_and(base_mask, selector_mask)
    return result


def extract_variables(expr):
    return {node.id for node in ast.walk(ast.parse(expr)) if isinstance(node, ast.Name)}
