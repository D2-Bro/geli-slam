"""ORB-SLAM3-style body-local pose increments and separate pose errors."""

import gtsam
import numpy as np

from gelslam.core.numpy_vi import adjoint


def split_pose_plus(pose, increment):
    """Use the pre-update R for p += R dp, then R *= Exp(dtheta)."""
    pose = np.asarray(pose, dtype=np.float64)
    increment = np.asarray(increment, dtype=np.float64)
    result = pose.copy()
    result[:3, 3] = pose[:3, 3] + pose[:3, :3] @ increment[3:6]
    result[:3, :3] = pose[:3, :3] @ gtsam.Rot3.Expmap(increment[:3]).matrix()
    return result


def split_pose_coordinates(anchor, pose):
    """Return [Log(R0.T R), R0.T (p-p0)], with no SE(3) log coupling."""
    relative_rotation = anchor[:3, :3].T @ pose[:3, :3]
    return np.concatenate((
        gtsam.Rot3.Logmap(gtsam.Rot3(relative_rotation)),
        anchor[:3, :3].T @ (pose[:3, 3] - anchor[:3, 3]),
    ))


def split_pose_coordinate_jacobian(anchor, pose):
    """Derivative of split coordinates under a split body-local increment."""
    coordinates = split_pose_coordinates(anchor, pose)
    J = np.zeros((6, 6))
    J[:3, :3] = gtsam.Rot3.LogmapDerivative(coordinates[:3])
    J[3:, 3:] = anchor[:3, :3].T @ pose[:3, :3]
    return J


def split_prior_factor(key, anchor, noise, state_T_observation=None):
    """Constrain an observation pose derived from an optional body offset."""
    anchor = np.asarray(anchor, dtype=np.float64).copy()
    offset = np.eye(4) if state_T_observation is None else np.asarray(state_T_observation).copy()
    observation_from_state = adjoint(np.linalg.inv(offset))

    def error(_factor, values, jacobians):
        pose = values.atPose3(key).matrix() @ offset
        if jacobians is not None:
            jacobians[0] = split_pose_coordinate_jacobian(anchor, pose) @ observation_from_state
        return split_pose_coordinates(anchor, pose)

    return gtsam.CustomFactor(noise, [key], error)


def split_between_factor(reference, target, measurement, noise, state_T_observation=None):
    """Compare relative observation poses, with Jacobians to body variables."""
    measurement = np.asarray(measurement, dtype=np.float64).copy()
    measurement_R_inv = measurement[:3, :3].T
    offset = np.eye(4) if state_T_observation is None else np.asarray(state_T_observation).copy()
    observation_from_state = adjoint(np.linalg.inv(offset))

    def error(_factor, values, jacobians):
        reference_pose = values.atPose3(reference).matrix() @ offset
        target_pose = values.atPose3(target).matrix() @ offset
        Rr, Rt = reference_pose[:3, :3], target_pose[:3, :3]
        relative = np.eye(4)
        relative[:3, :3] = Rr.T @ Rt
        relative[:3, 3] = Rr.T @ (target_pose[:3, 3] - reference_pose[:3, 3])
        residual = split_pose_coordinates(measurement, relative)
        if jacobians is not None:
            Jr = gtsam.Rot3.LogmapDerivative(residual[:3])
            x, y, z = relative[:3, 3]
            skew = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
            reference_J = np.zeros((6, 6))
            target_J = np.zeros((6, 6))
            reference_J[:3, :3] = -Jr @ relative[:3, :3].T
            reference_J[3:, :3] = measurement_R_inv @ skew
            reference_J[3:, 3:] = -measurement_R_inv
            target_J[:3, :3] = Jr
            target_J[3:, 3:] = measurement_R_inv @ relative[:3, :3]
            jacobians[0] = reference_J @ observation_from_state
            jacobians[1] = target_J @ observation_from_state
        return residual

    return gtsam.CustomFactor(noise, [reference, target], error)


def solve_gaussian_qr(graph):
    """Use sparse sequential QR with the Python-exposed elimination function.

    Parallel elimination may invoke that Python callback from worker threads
    while its caller holds the GIL. Sequential elimination avoids this wait.
    """
    ordering = gtsam.Ordering.ColamdGaussianFactorGraph(graph)
    return graph.eliminateSequential(ordering, gtsam.EliminateQR).optimize()


def _condition_fixed_keys(linear, current, widths, fixed_keys):
    """Condition each Gaussian row on delta_fixed=0 without a finite prior."""
    conditioned = gtsam.GaussianFactorGraph()
    for index in range(linear.size()):
        factor = linear.at(index)
        if factor is None:
            continue
        keys = list(factor.keys())
        if not fixed_keys.intersection(keys):
            conditioned.add(factor)
            continue
        if all(key in fixed_keys for key in keys):
            continue
        A, rhs = gtsam.JacobianFactor(factor).jacobian()
        blocks, column = {}, 0
        for key in keys:
            width = widths[key]
            if key not in fixed_keys:
                blocks[key] = A[:, column:column + width]
            column += width
        noise = gtsam.noiseModel.Unit.Create(len(rhs))
        if len(blocks) <= 3:
            args = [item for key, block in blocks.items() for item in (key, block)]
            conditioned.add(gtsam.JacobianFactor(*args, rhs, noise))
        else:
            # The Python JacobianFactor constructors expose at most three
            # keys. CustomFactor can construct arbitrary Gaussian row blocks.
            def rows(_factor, _values, jacobians):
                if jacobians is not None:
                    for slot, block in enumerate(blocks.values()):
                        jacobians[slot] = block
                return -rhs

            conditioned.add(gtsam.CustomFactor(noise, list(blocks), rows).linearize(current))
    return conditioned


def optimize_split_pose_graph(graph, initial, max_iterations=100, variable_types=None,
                              fixed_keys=()):
    """Sparse GTSAM QR steps with LM damping and explicit split retraction.

    Native Pose3 optimizers use Values.retract(), whose installed chart uses
    the SE(3) exponential. Apply accepted increments ourselves instead.
    Damping follows each variable's information diagonal, so translation
    and rotation with different units do not share an absolute scale.
    Fixed keys retain their supplied values exactly. Their increment columns
    are removed before QR, while their observations still constrain free keys.
    """
    current = gtsam.Values(initial)
    all_keys = list(current.keys())
    fixed_keys = set(fixed_keys)
    if not fixed_keys.issubset(all_keys):
        raise ValueError("fixed keys must exist in the initial values")
    keys = [key for key in all_keys if key not in fixed_keys]
    types = {key: "pose" for key in all_keys} if variable_types is None else variable_types
    widths = {key: (len(current.atVector(key)) if types[key] == "vector" else 6)
              for key in all_keys}
    cost = float(graph.error(current))
    if not np.isfinite(cost):
        raise ValueError("non-finite split pose graph cost")
    if not keys:
        return current
    damping = 1e-5
    for _iteration in range(max_iterations):
        if cost <= 1e-16:
            break
        linear = graph.linearize(current)
        if fixed_keys:
            linear = _condition_fixed_keys(linear, current, widths, fixed_keys)
        diagonals = {key: np.zeros(widths[key]) for key in keys}
        for index in range(linear.size()):
            factor = linear.at(index)
            if factor is None:
                continue
            A, _ = gtsam.JacobianFactor(factor).jacobian()
            column = 0
            for key in factor.keys():
                width = widths[key]
                diagonals[key] += np.sum(A[:, column:column + width]**2, axis=0)
                column += width
        accepted = False
        for _attempt in range(12):
            damped = gtsam.GaussianFactorGraph()
            for index in range(linear.size()):
                factor = linear.at(index)
                if factor is not None:
                    damped.add(factor)
            for key in keys:
                scale = np.sqrt(np.maximum(diagonals[key], 1e-12) * damping)
                damped.add(gtsam.JacobianFactor(
                    key, np.diag(scale), np.zeros(widths[key]),
                    gtsam.noiseModel.Isotropic.Sigma(widths[key], 1.0),
                ))
            solution = solve_gaussian_qr(damped)
            candidate = gtsam.Values(current)
            for key in keys:
                step = np.asarray(solution.at(key))
                if not np.all(np.isfinite(step)):
                    raise ValueError("non-finite split pose graph increment")
                if types[key] == "pose":
                    candidate.update(key, gtsam.Pose3(split_pose_plus(
                        current.atPose3(key).matrix(), step
                    )))
                elif types[key] == "vector":
                    candidate.update(key, current.atVector(key) + step)
                else:
                    bias = current.atConstantBias(key).vector() + step
                    candidate.update(key, gtsam.imuBias.ConstantBias(bias[:3], bias[3:]))
            candidate_cost = float(graph.error(candidate))
            if np.isfinite(candidate_cost) and candidate_cost < cost:
                improvement = cost - candidate_cost
                current, cost = candidate, candidate_cost
                damping = max(damping / 10., 1e-12)
                accepted = True
                break
            damping *= 10.
        if not accepted or improvement <= 1e-8 * max(cost, 1.0):
            break
    return current
