import numpy as np
import scipy
from scipy.spatial import ConvexHull

import torch


def sample_labels(size: int, lim: list, seed: int = None) -> np.ndarray:
    """What's meaning of sample_labels?"""
    if seed:
        np.random.seed(seed)
    all_sands = np.arange(*lim)  # lim是只含有两个int的列表[lo, hi]
    # 不放回简单随机抽样
    sampled_sands = np.sort(np.random.choice(all_sands, size, replace=False))
    return sampled_sands


def project(tensor: torch.tensor, dim: int):
    return torch.max(tensor, dim=dim).values


class Circumsphere:
    """Copied from GitHub at https://github.com/shrx/mbsc.git, not my original.
    """
    @classmethod
    def fit(cls, array):
        """Compute exact minimum bounding sphere of a 3D point cloud (or a
        triangular surface mesh) using Welzl's algorithm.

        - X     : M-by-3 list of point co-ordinates or a triangular surface
                            mesh specified as a TriRep object.
        - R     : radius of the sphere.
        - C     : 1-by-3 vector specifying the centroid of the sphere.
        - Xb    : subset of X, listing K-by-3 list of point coordinates from
                            which R and C were computed. See function titled
                            'FitSphere2Points' for more info.

        REREFERENCES:
        [1] Welzl, E. (1991), 'Smallest enclosing disks (balls and ellipsoids)',
            Lecture Notes in Computer Science, Vol. 555, pp. 359-370

        Matlab code author: Anton Semechko (a.semechko@gmail.com)
        Date: Dec.2014"""

        # Get the convex hull of the point set
        hull = ConvexHull(array)
        hull_array = array[hull.vertices]
        hull_array = np.unique(hull_array, axis=0)
        # print(len(hull_array))

        # Randomly permute the point set
        hull_array = np.random.permutation(hull_array)

        if len(hull_array) <= 4:
            R, C = cls.fit_base(hull_array)
            return R, C, hull_array

        elif len(hull_array) < 1000:
            # try:
            R, C, _ = cls.B_min_sphere(hull_array, [])

            # Coordiantes of the points used to compute parameters of the
            # minimum bounding sphere
            D = np.sum(np.square(hull_array - C), axis=1)
            idx = np.argsort(D - R**2)
            D = D[idx]
            Xb = hull_array[idx[:5]]
            D = D[:5]
            Xb = Xb[D < 1E-6]
            idx = np.argsort(Xb[:, 0])
            Xb = Xb[idx]
            return R, C, Xb
            # except:
            #raise Exception
        else:
            M = len(hull_array)
            dM = min([M // 4, 300])
        # unnecessary ?
        #		res = M % dM
        #		n = np.ceil(M/dM)
        #		idx = dM * np.ones((1, n))
        #		if res > 0:
        #			idx[-1] = res
        #
        #		if res <= 0.25 * dM:
        #			idx[n-2] = idx[n-2] + idx[n-1]
        #			idx = idx[:-1]
        #			n -= 1

            hull_array = np.array_split(hull_array, dM)
            Xb = np.empty([0, 3])
            for i in range(len(hull_array)):
                R, C, Xi = cls.B_min_sphere(
                    np.vstack([Xb, hull_array[i]]), [])

                # 40 points closest to the sphere
                D = np.abs(np.sqrt(np.sum((Xi - C)**2, axis=1)) - R)
                idx = np.argsort(D, axis=0)
                Xb = Xi[idx[:40]]

            D = np.sort(D, axis=0)[:4]
            # print(Xb)
            # print(D)
            #print(np.where(D/R < 1e-3)[0])
            Xb = np.take(Xb, np.where(D/R < 1e-3)[0], axis=0)
            Xb = np.sort(Xb, axis=0)
            # print(Xb)

            return R, C, Xb

    @classmethod
    def fit_base(cls, array):
        """Fit a sphere to a set of 2, 3, or at most 4 points in 3D space. Note that
        point configurations with 3 collinear or 4 coplanar points do not have 
        well-defined solutions (i.e., they lie on spheres with inf radius).

        - X     : M-by-3 array of point coordinates, where M<=4.
        - R     : radius of the sphere. R=Inf when the sphere is undefined, as 
                    specified above.
        - C     : 1-by-3 vector specifying the centroid of the sphere. 
                    C=nan(1,3) when the sphere is undefined, as specified above.

        Matlab code author: Anton Semechko (a.semechko@gmail.com)
        Date: Dec.2014"""

        N = len(array)

        if N > 4:
            print('Input must a N-by-3 array of point coordinates, with N<=4')
            return

        # Empty set
        elif N == 0:
            R = np.nan
            C = np.full(3, np.nan)
            return R, C

        # A single point
        elif N == 1:
            R = 0.
            C = array[0]
            return R, C

        # Line segment
        elif N == 2:
            R = np.linalg.norm(array[1] - array[0]) / 2
            C = np.mean(array, axis=0)
            return R, C

        else:  # 3 or 4 points
            # Remove duplicate vertices, if there are any
            uniq, index = np.unique(array, axis=0, return_index=True)
            array_nd = uniq[index.argsort()]
            if not np.array_equal(array, array_nd):
                print("found duplicate")
                print(array_nd)
                R, C = cls.fit_base(array_nd)
                return R, C

            tol = 0.01  # collinearity/co-planarity threshold (in degrees)
            if N == 3:
                # Check for collinearity
                D12 = array[1] - array[0]
                D12 = D12 / np.linalg.norm(D12)
                D13 = array[2] - array[0]
                D13 = D13 / np.linalg.norm(D13)

                chk = np.clip(np.abs(np.dot(D12, D13)), 0., 1.)
                if np.arccos(chk)/np.pi*180 < tol:
                    R = np.inf
                    C = np.full(3, np.nan)
                    return R, C

                # Make plane formed by the points parallel with the xy-plane
                n = np.cross(D13, D12)
                n = n / np.linalg.norm(n)
                ##print("n", n)
                r = np.cross(n, np.array([0, 0, 1]))
                if np.linalg.norm(r) != 0:
                    # Euler rotation vector
                    r = np.arccos(n[2]) * r / np.linalg.norm(r)
                ##print("r", r)
                Rmat = scipy.linalg.expm(np.array([
                    [0., -r[2], r[1]],
                    [r[2], 0., -r[0]],
                    [-r[1], r[0], 0.]
                ]))
                ##print("Rmat", Rmat)
                #Xr = np.transpose(Rmat*np.transpose(array))
                Xr = np.transpose(np.dot(Rmat, np.transpose(array)))
                ##print("Xr", Xr)

                # Circle centroid
                x = Xr[:, :2]
                A = 2 * (x[1:] - np.full(2, x[0]))
                b = np.sum(
                    (np.square(x[1:]) - np.square(np.full(2, x[0]))), axis=1)
                C = np.transpose(np.linalg.solve(A, b))

                # Circle radius
                R = np.sqrt(np.sum(np.square(x[0] - C)))

                # Rotate centroid back into the original frame of reference
                C = np.append(C, [np.mean(Xr[:, 2])], axis=0)
                C = np.transpose(np.dot(np.transpose(Rmat), C))
                return R, C

            # If we got to this point then we have 4 unique, though possibly co-linear
            # or co-planar points.
            else:
                # Check if the the points are co-linear
                D12 = array[1] - array[0]
                D12 = D12 / np.linalg.norm(D12)
                D13 = array[2] - array[0]
                D13 = D13 / np.linalg.norm(D13)
                D14 = array[3] - array[0]
                D14 = D14 / np.linalg.norm(D14)

                chk1 = np.clip(np.abs(np.dot(D12, D13)), 0., 1.)
                chk2 = np.clip(np.abs(np.dot(D12, D14)), 0., 1.)
                if np.arccos(chk1)/np.pi*180 < tol or np.arccos(chk2)/np.pi*180 < tol:
                    R = np.inf
                    C = np.full(3, np.nan)
                    return R, C

                # Check if the the points are co-planar
                n1 = np.linalg.norm(np.cross(D12, D13))
                n2 = np.linalg.norm(np.cross(D12, D14))

                chk = np.clip(np.abs(np.dot(n1, n2)), 0., 1.)
                if np.arccos(chk)/np.pi*180 < tol:
                    R = np.inf
                    C = np.full(3, np.nan)
                    return R, C

                # Centroid of the sphere
                A = 2 * (array[1:] - np.full(len(array)-1, array[0]))
                b = np.sum(
                    (np.square(array[1:]) - np.square(np.full(len(array)-1, array[0]))), axis=1)
                C = np.transpose(np.linalg.solve(A, b))

                # Radius of the sphere
                R = np.sqrt(np.sum(np.square(array[0] - C), axis=0))

                return R, C

    @classmethod
    def B_min_sphere(cls, P, B):
        eps = 1E-6
        if len(B) == 4 or len(P) == 0:
            R, C = cls.fit_base(B)  # fit sphere to boundary points
            return R, C, P

        # Remove the last (i.e., end) point, p, from the list
        P_new = P[:-1].copy()
        p = P[-1].copy()

        # Check if p is on or inside the bounding sphere. If not, it must be
        # part of the new boundary.
        R, C, P_new = cls.B_min_sphere(P_new, B)
        if np.isnan(R) or np.isinf(R) or R < eps:
            chk = True
        else:
            chk = np.linalg.norm(p - C) > (R + eps)

        if chk:
            if len(B) == 0:
                B = np.array([p])
            else:
                B = np.array(np.insert(B, 0, p, axis=0))
            R, C, _ = cls.B_min_sphere(P_new, B)
            P = np.insert(P_new.copy(), 0, p, axis=0)
        return R, C, P


if __name__ == '__main__':
    pass
