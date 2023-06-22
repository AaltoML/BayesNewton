import jax.numpy as np
from jax import vmap
from jax.scipy.linalg import cho_factor, cho_solve
from jax.scipy.linalg import solve as jsc_solve
from .utils import mvn_logpdf, solve, transpose, inv, inv_vmap
from jax.lax import scan, associative_scan
import math

INV2PI = (2 * math.pi) ** -1


def get_diag_and_offdiag_components(num_latents, zeros, i, noise_cov):
    temp_vec = zeros.at[i].add(1.)
    temp_mat = temp_vec.reshape(num_latents, num_latents)
    return np.kron(np.diag(noise_cov), temp_mat)  # block-diag


def blocktensor_to_blockdiagmatrix(blocktensor):
    """
    Convert [N, D, D] tensor to [ND, ND] block-diagonal matrix
    """
    N = blocktensor.shape[0]
    D = blocktensor.shape[1]
    diag_and_offdiag_components = vmap(get_diag_and_offdiag_components, in_axes=(None, None, 0, 1))(
        D, np.zeros(D ** 2), np.arange(D ** 2), blocktensor.reshape(N, -1)
    )
    return np.sum(diag_and_offdiag_components, axis=0)


def get_blocks(blockdiagmatrix, D, i):
    return blockdiagmatrix[0+D*i:D+D*i, 0+D*i:D+D*i]


@vmap
def get_3d_off_diag(offdiag_elems):
    return np.sum(
        offdiag_elems[:, None, None]
        * np.array([[[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]],
                    [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]]]),
        axis=0
    )


def blockdiagmatrix_to_blocktensor(blockdiagmatrix, N, D):
    """
    Convert [ND, ND] block-diagonal matrix to [N, D, D] tensor
    Code from https://stackoverflow.com/questions/10831417/extracting-diagonal-blocks-from-a-numpy-array
    """
    return np.array([blockdiagmatrix[i*D:(i+1)*D,i*D:(i+1)*D] for i in range(N)])


def gaussian_conditional(kernel, y, noise_cov, X, X_star=None):
    """
    Compute the GP posterior / predictive distribution using standard Gaussian identities
    :param kernel: an instantiation of the kernel class
    :param y: observations [N, 1]
    :param noise_cov: observation noise covariance [N, 1]
    :param X: training inputs [N, D]
    :param X_star: test inputs [N*, D]
    :return:
        mean: posterior mean [N, 1]
        covariance: posterior covariance [N, N]
    """
    Kff = kernel(X, X)
    if X_star is None:  # inference / learning
        Kfs = Kff
        Kss = Kff
    else:  # prediction
        Kfs = kernel(X, X_star)
        Kss = kernel(X_star, X_star)

    # Ky = Kff + np.diag(np.squeeze(noise_cov))  # single-latent version
    noise_cov_block_diag = blocktensor_to_blockdiagmatrix(noise_cov)  # multi-latent version
    Ky = Kff + noise_cov_block_diag
    # ---- compute approximate posterior using standard Gaussian conditional formula ----
    Kfs_iKy = solve(Ky, Kfs).T
    # mean = Kfs_iKy @ diag(y)
    mean = Kfs_iKy @ y.reshape(-1, 1)
    covariance = Kss - Kfs_iKy @ Kfs
    return mean, covariance


def sparse_gaussian_conditional(kernel, nat1lik, nat2lik, X, Z):
    """
    Compute q(u)
    :param kernel: an instantiation of the kernel class
    :param nat1lik: likelihood first natural parameter [N, 1]
    :param nat2lik: likelihood noise precision [N, 1]
    :param X: training inputs [N, D]
    :param Z: inducing inputs [N*, D]
    :return:
        mean: posterior mean [N, 1]
        covariance: posterior covariance [N, N]
    """
    Kuf = kernel(Z, X)
    Kuu = kernel(Z, Z)
    # nat2prior = inv(Kuu)
    Wuf = solve(Kuu, Kuf)  # conditional mapping, Kuu^-1 Kuf

    nat1lik_fullrank = Wuf @ nat1lik.reshape(-1, 1)
    # nat2lik_fullrank = Wuf @ np.diag(np.squeeze(nat2lik)) @ transpose(Wuf)
    nat2lik_block_diag = blocktensor_to_blockdiagmatrix(nat2lik)  # multi-latent version
    nat2lik_fullrank = Wuf @ nat2lik_block_diag @ transpose(Wuf)

    # nat1post = nat1lik_fullrank  # prior nat1 is zero
    # nat2post = nat2prior + nat2lik_fullrank

    # covariance = inv(nat2post)
    # mean = covariance @ nat1post

    likcov = inv(nat2lik_fullrank)
    likmean = likcov @ nat1lik_fullrank
    Ky = Kuu + likcov
    # ---- compute approximate posterior using standard Gaussian conditional formula ----
    Kuu_iKy = solve(Ky, Kuu).T
    mean = Kuu_iKy @ likmean
    covariance = Kuu - Kuu_iKy @ Kuu

    return mean, covariance


def sparse_conditional_post_to_data(kernel, post_mean, post_cov, X, Z):
    """
    Compute int p(f|u) q(u) du
    :param kernel: an instantiation of the kernel class
    :param post_mean: posterior mean [M, 1]
    :param post_cov: posterior covariance [M, M]
    :param X: training inputs [N, D]
    :param Z: inducing inputs [N*, D]
    :return:
        mean: posterior mean [N, 1]
        covariance: posterior covariance [N, N]
    """
    X = X.reshape(-1, Z.shape[1])
    Kff = kernel(X, X)
    Kuf = kernel(Z, X)
    Kuu = kernel(Z, Z)
    Wuf = solve(Kuu, Kuf)  # conditional mapping, Kuu^-1 Kuf
    Qff = transpose(Kuf) @ Wuf  # Kfu Kuu^-1 Kuf

    conditional_cov = Kff - Qff

    mean_f = transpose(Wuf) @ post_mean.reshape(-1, 1)
    cov_f = conditional_cov + transpose(Wuf) @ post_cov @ Wuf

    return mean_f, cov_f


def process_noise_covariance(A, Pinf):
    Q = Pinf - A @ Pinf @ transpose(A)
    return Q


def _sequential_kf(As, Qs, H, ys, noise_covs, m0, P0, masks, return_predict=False):

    def body(carry, inputs):
        y, A, Q, obs_cov, mask = inputs
        m, P, ell = carry
        m_ = A @ m
        P_ = A @ P @ A.T + Q

        obs_mean = H @ m_
        HP = H @ P_
        S = HP @ H.T + obs_cov

        ell_n = mvn_logpdf(y, obs_mean, S, mask)
        ell = ell + ell_n

        K = solve(S, HP).T
        m = m_ + K @ (y - obs_mean)
        P = P_ - K @ HP
        if return_predict:
            return (m, P, ell), (m_, P_)
        else:
            return (m, P, ell), (m, P)

    (_, _, loglik), (fms, fPs) = scan(f=body,
                                      init=(m0, P0, 0.),
                                      xs=(ys, As, Qs, noise_covs, masks))
    return loglik, fms, fPs


def parallel_filtering_element_(A, Q, H, noise_cov, y):
    HQ, HA = H @ Q, H @ A  # pre-compute intermediates
    S = HQ @ H.T + noise_cov  # H Q H.T + R

    SinvH = solve(S, H)  # S^{-1} H

    K = Q @ SinvH.T  # Q H.T S^{-1}
    AA = A - K @ HA  # A - K H A
    b = K @ y  # K y
    C = Q - K @ HQ  # Q - K H Q

    SinvHA = (SinvH @ A).T  # A.T H.T S^{-1}
    eta = SinvHA @ y  # A.T H.T S^{-1} y
    J = SinvHA @ HA  # A.T H.T S^{-1} H A
    return AA, b, C, J, eta


parallel_filtering_element = vmap(parallel_filtering_element_, in_axes=(0, 0, None, 0, 0))


@vmap
def parallel_filtering_operator(elem1, elem2):
    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2

    C1inv = inv(C1)
    temp = solve(C1inv + J2, C1inv)  # we should avoid inverting non-PSD matrices here
    A2temp = A2 @ temp
    AA = A2temp @ A1
    b = A2temp @ (b1 + C1 @ eta2) + b2
    C = A2temp @ C1 @ A2.T + C2

    A1temp = A1.T @ temp.T  # re-use previous solve
    eta = A1temp @ (eta2 - J2 @ b1) + eta1
    J = A1temp @ J2 @ A1 + J1

    return AA, b, C, J, eta


def make_associative_filtering_elements(As, Qs, H, ys, noise_covs, m0, P0):
    Qs = Qs.at[0].set(P0)  # first element requires different initialisation
    AA, b, C, J, eta = parallel_filtering_element(As, Qs, H, noise_covs, ys)
    # modify initial b to account for m0 (not needed if m0=zeros)
    S = H @ Qs[0] @ H.T + noise_covs[0]
    K0 = solve(S, H @ Qs[0]).T
    b = b.at[0].add(m0 - K0 @ H @ m0)
    return AA, b, C, J, eta


@vmap
def vmap_mvn_logpdf(*args, **kwargs):
    return mvn_logpdf(*args, **kwargs)


def _parallel_kf(As, Qs, H, ys, noise_covs, m0, P0, masks, return_predict=False):

    # perform parallel filtering
    initial_elements = make_associative_filtering_elements(As, Qs, H, ys, noise_covs, m0, P0)
    final_elements = associative_scan(parallel_filtering_operator, initial_elements)
    fms, fPs = final_elements[1], final_elements[2]

    # now compute the log likelihood
    mpredict = As @ np.concatenate([m0[None], fms[:-1]])
    Ppredict = As @ np.concatenate([P0[None], fPs[:-1]]) @ transpose(As) + Qs

    loglik = np.sum(vmap_mvn_logpdf(ys, H @ mpredict, H @ Ppredict @ H.T + noise_covs, masks))

    if return_predict:
        return loglik, mpredict, Ppredict
    else:
        return loglik, fms, fPs


def kalman_filter(dt, kernel, y, noise_cov, mask=None, parallel=False, return_predict=False):
    """
    Run the Kalman filter to get p(fₙ|y₁,...,yₙ).
    Assumes a heteroscedastic Gaussian observation model, i.e. var is vector valued
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, D, 1]
    :param noise_cov: observation noise covariances [N, D, D]
    :param mask: boolean mask for the observations (to indicate missing data locations) [N, D, 1]
    :param parallel: flag to switch between parallel and sequential implementation of Kalman filter
    :param return_predict: flag whether to return predicted state, rather than updated state
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """
    if mask is None:
        mask = np.zeros_like(y, dtype=bool)
    Pinf = kernel.stationary_covariance()
    minf = np.zeros([Pinf.shape[0], 1])

    As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    H = kernel.measurement_model()

    if parallel:
        ell, means, covs = _parallel_kf(As, Qs, H, y, noise_cov, minf, Pinf, mask, return_predict=return_predict)
    else:
        ell, means, covs = _sequential_kf(As, Qs, H, y, noise_cov, minf, Pinf, mask, return_predict=return_predict)
    return ell, (means, covs)


def _sequential_rts(fms, fPs, As, Qs, H, return_full):

    def body(carry, inputs):
        fm, fP, A, Q = inputs
        sm, sP = carry

        pm = A @ fm
        AfP = A @ fP
        pP = AfP @ A.T + Q

        C = solve(pP, AfP).T

        sm = fm + C @ (sm - pm)
        sP = fP + C @ (sP - pP) @ C.T
        if return_full:
            return (sm, sP), (sm, sP, C)
        else:
            return (sm, sP), (H @ sm, H @ sP @ H.T, C)

    _, (sms, sPs, gains) = scan(f=body,
                                init=(fms[-1], fPs[-1]),
                                xs=(fms, fPs, As, Qs),
                                reverse=True)
    return sms, sPs, gains


def last_parallel_smoothing_element(m, P):
    return np.zeros_like(P), m, P


@vmap
def parallel_smoothing_element(A, Q, m, P):
    Pp = A @ P @ transpose(A) + Q

    E = transpose(solve(Pp, A @ P))
    g = m - E @ A @ m
    L = P - E @ Pp @ transpose(E)
    return E, g, L


@vmap
def parallel_smoothing_operator(elem1, elem2):
    E1, g1, L1 = elem1
    E2, g2, L2 = elem2
    E = E2 @ E1
    g = E2 @ g1 + g2
    L = E2 @ L1 @ transpose(E2) + L2
    return E, g, L


def _parallel_rts(fms, fPs, As, Qs, H, return_full):

    # build the associative smoothing elements
    smoothing_elems = parallel_smoothing_element(As, Qs, fms, fPs)
    gains = smoothing_elems[0]
    last_elems = last_parallel_smoothing_element(fms[-1], fPs[-1])
    initial_elements = tuple(np.append(gen_es[:-1], np.expand_dims(last_e, 0), axis=0)
                             for gen_es, last_e in zip(smoothing_elems, last_elems))

    # run the parallel smoother
    final_elements = associative_scan(parallel_smoothing_operator, initial_elements, reverse=True)  # note vmap
    sms, sPs = final_elements[1], final_elements[2]

    if return_full:
        return sms, sPs, gains
    else:
        return H @ sms, H @ sPs @ H.T, gains


def rauch_tung_striebel_smoother(dt, kernel, filter_mean, filter_cov, return_full=False, parallel=False):
    """
    Run the RTS smoother to get p(fₙ|y₁,...,y_N),
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param filter_mean: the intermediate distribution means computed during filtering [N, state_dim, 1]
    :param filter_cov: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
    :param return_full: a flag determining whether to return the full state distribution or just the function(s)
    :param parallel: flag to switch between parallel and sequential implementation of smoother
    :return:
        smoothed_mean: the posterior marginal means [N, obs_dim]
        smoothed_var: the posterior marginal variances [N, obs_dim]
    """
    Pinf = kernel.stationary_covariance()

    As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    H = kernel.measurement_model()

    if parallel:
        means, covs, gains = _parallel_rts(filter_mean, filter_cov, As, Qs, H, return_full)
    else:
        means, covs, gains = _sequential_rts(filter_mean, filter_cov, As, Qs, H, return_full)
    return means, covs, gains


def kalman_filter_pairs(dt, kernel, y, noise_cov, mask=None, parallel=False):
    """
    A Kalman filter over pairs of states, in which y is [2state_dim, 1] and noise_cov is [2state_dim, 2state_dim]
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, 2state_dim, 1]
    :param noise_cov: observation noise covariances [N, 2state_dim, 2state_dim]
    :param mask: boolean mask for the observations (to indicate missing data locations) [N, 2state_dim, 1]
    :param parallel: flag to switch between parallel and sequential implementation of Kalman filter
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: marginal state filtering means [N, state_dim, 1]
        covs: marginal state filtering covariances [N, state_dim, state_dim]
    """
    if mask is None:
        mask = np.zeros_like(y, dtype=bool)

    Pinf = kernel.stationary_covariance()
    state_dim = Pinf.shape[0]
    minf = np.zeros([state_dim, 1])
    zeros = np.zeros([state_dim, state_dim])
    Pinfpair = np.block([[Pinf,  zeros],
                         [zeros, Pinf]])
    minfpair = np.block([[minf],
                         [minf]])

    As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)

    def construct_pair(A, Q):
        Apair = np.block([[zeros, np.eye(state_dim)],
                          [zeros, A]])
        Qpair = np.block([[1e-32 * np.eye(state_dim), zeros],  # jitter avoids numerical errors in parallel filter init
                          [zeros,                     Q]])
        return Apair, Qpair

    Apairs, Qpairs = vmap(construct_pair)(As, Qs)
    H = np.eye(2 * state_dim)

    if parallel:
        ell, means, covs = _parallel_kf(Apairs, Qpairs, H, y, noise_cov, minfpair, Pinfpair, mask)
    else:
        ell, means, covs = _sequential_kf(Apairs, Qpairs, H, y, noise_cov, minfpair, Pinfpair, mask)
    return ell, (means[1:, :state_dim], covs[1:, :state_dim, :state_dim])


def _sequential_kf_mf(As, Qs, H, ys, noise_covs, m0, P0, masks, block_index):

    def build_block_diag(P_blocks):
        P = Pzeros.at[block_index].add(P_blocks.flatten())
        return P

    def get_block_mean(m):
        return m.reshape(num_latents, sub_state_dim, 1)

    def get_block_cov(P):
        return P[block_index].reshape(num_latents, sub_state_dim, sub_state_dim)

    num_latents = m0.shape[0]
    sub_state_dim = m0.shape[1]
    state_dim = num_latents * sub_state_dim
    Pzeros = np.zeros([state_dim, state_dim])

    def body(carry, inputs):
        y, A, Q, obs_cov, mask = inputs
        m, P, ell = carry
        m = (A @ m).reshape(-1, 1)
        P = build_block_diag(A @ P @ transpose(A) + Q)

        obs_mean = H @ m
        HP = H @ P
        S = HP @ H.T + obs_cov

        ell_n = mvn_logpdf(y, obs_mean, S, mask)
        ell = ell + ell_n

        K = solve(S, HP).T
        m = get_block_mean(m + K @ (y - obs_mean))
        P = get_block_cov(P - K @ HP)
        return (m, P, ell), (m, P)

    (_, _, loglik), (fms, fPs) = scan(f=body,
                                      init=(m0, P0, 0.),
                                      xs=(ys, As, Qs, noise_covs, masks))
    return loglik, fms, fPs


def parallel_filtering_element_mf_(A, Q, H, noise_cov, y, block_index):

    def build_block_diag(P_blocks):
        P = Pzeros.at[block_index].add(P_blocks.flatten())
        return P

    def get_block_mean(m):
        return m.reshape(num_latents, sub_state_dim, 1)

    def get_block_cov(P):
        return P[block_index].reshape(num_latents, sub_state_dim, sub_state_dim)

    num_latents = A.shape[0]
    sub_state_dim = A.shape[1]
    state_dim = num_latents * sub_state_dim
    Pzeros = np.zeros([state_dim, state_dim])

    Q, A = build_block_diag(Q), build_block_diag(A)
    HQ, HA = H @ Q, H @ A  # pre-compute intermediates
    S = HQ @ H.T + noise_cov  # H Q H.T + R

    SinvH = solve(S, H)  # S^{-1} H

    K = Q @ SinvH.T  # Q H.T S^{-1}
    AA = get_block_cov(A - K @ HA)  # A - K H A
    b = get_block_mean(K @ y)  # K y
    C = get_block_cov(Q - K @ HQ)  # Q - K H Q

    SinvHA = (SinvH @ A).T  # A.T H.T S^{-1}
    eta = get_block_mean(SinvHA @ y)  # A.T H.T S^{-1} y
    J = get_block_cov(SinvHA @ HA)  # A.T H.T S^{-1} H A
    return AA, b, C, J, eta


parallel_filtering_element_mf = vmap(parallel_filtering_element_mf_, in_axes=(0, 0, None, 0, 0, None))


@vmap
def parallel_filtering_operator_mf(elem1, elem2):
    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2

    C1inv = inv_vmap(C1)
    temp = solve(C1inv + J2, C1inv)  # we should avoid inverting non-PSD matrices here
    A2temp = A2 @ temp
    AA = A2temp @ A1
    b = A2temp @ (b1 + C1 @ eta2) + b2
    C = A2temp @ C1 @ transpose(A2) + C2

    A1temp = transpose(A1) @ transpose(temp)  # re-use previous solve
    eta = A1temp @ (eta2 - J2 @ b1) + eta1
    J = A1temp @ J2 @ A1 + J1

    return AA, b, C, J, eta


def make_associative_filtering_elements_mf(As, Qs, H, ys, noise_covs, m0, P0, block_index):

    def build_block_diag(P_blocks):
        P = Pzeros.at[block_index].add(P_blocks.flatten())
        return P

    def get_block_mean(m):
        return m.reshape(num_latents, sub_state_dim, 1)

    num_latents = As.shape[1]
    sub_state_dim = As.shape[2]
    state_dim = num_latents * sub_state_dim
    Pzeros = np.zeros([state_dim, state_dim])

    Qs = Qs.at[0].set(P0)  # first element requires different initialisation
    AA, b, C, J, eta = parallel_filtering_element_mf(As, Qs, H, noise_covs, ys, block_index)
    # modify initial b to account for m0 (not needed if m0=zeros)
    m0 = m0.reshape(-1, 1)
    Qs0 = build_block_diag(Qs[0])
    S = H @ Qs0 @ H.T + noise_covs[0]
    K0 = solve(S, H @ Qs0).T
    b = b.at[0].add(get_block_mean(m0 - K0 @ (H @ m0)))
    return AA, b, C, J, eta


def _parallel_kf_mf(As, Qs, H, ys, noise_covs, m0, P0, masks, block_index):

    @vmap
    def build_block_diag(P_blocks):
        P = Pzeros.at[block_index].add(P_blocks.flatten())
        return P

    def build_mean(m):
        return m.reshape(num_time_steps, state_dim, 1)

    num_time_steps = As.shape[0]
    num_latents = As.shape[1]
    sub_state_dim = As.shape[2]
    state_dim = num_latents * sub_state_dim
    Pzeros = np.zeros([state_dim, state_dim])

    # perform parallel filtering
    initial_elements = make_associative_filtering_elements_mf(As, Qs, H, ys, noise_covs, m0, P0, block_index)
    final_elements = associative_scan(parallel_filtering_operator_mf, initial_elements)
    fms, fPs = final_elements[1], final_elements[2]

    # now compute the log likelihood
    mpredict = build_mean(As @ np.concatenate([m0[None], fms[:-1]]))
    Ppredict = build_block_diag(As @ np.concatenate([P0[None], fPs[:-1]]) @ transpose(As) + Qs)

    loglik = np.sum(vmap_mvn_logpdf(ys, H @ mpredict, H @ Ppredict @ H.T + noise_covs, masks))

    return loglik, fms, fPs


def kalman_filter_meanfield(dt, kernel, y, noise_cov, mask=None, parallel=False, block_index=None):
    """
    Run the Kalman filter to get p(fₙ|y₁,...,yₙ) with a mean-field approximation for the latent components.
    Assumes a heteroscedastic Gaussian observation model, i.e. var is vector valued
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, D, 1]
    :param noise_cov: observation noise covariances [N, D, D]
    :param mask: boolean mask for the observations (to indicate missing data locations) [N, D, 1]
    :param parallel: flag to switch between parallel and sequential implementation of Kalman filter
    :param block_index: mean-field block indices required to build the block-diagonal model matrices from the blocks
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """
    if mask is None:
        mask = np.zeros_like(y, dtype=bool)
    Pinf = kernel.stationary_covariance_meanfield()
    minf = np.zeros([Pinf.shape[0], Pinf.shape[1], 1])

    As = vmap(kernel.state_transition_meanfield)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    H = kernel.measurement_model()

    if parallel:
        ell, means, covs = _parallel_kf_mf(As, Qs, H, y, noise_cov, minf, Pinf, mask, block_index)
    else:
        ell, means, covs = _sequential_kf_mf(As, Qs, H, y, noise_cov, minf, Pinf, mask, block_index)
    return ell, (means, covs)


def _sequential_rts_mf(fms, fPs, As, Qs, H, return_full, block_index):

    def build_block_diag(P_blocks):
        P = Pzeros.at[block_index].add(P_blocks.flatten())
        return P

    num_latents = fms.shape[1]
    sub_state_dim = fms.shape[2]
    state_dim = num_latents * sub_state_dim
    Pzeros = np.zeros([state_dim, state_dim])

    def body(carry, inputs):
        fm, fP, A, Q = inputs
        sm, sP = carry

        pm = A @ fm
        AfP = A @ fP
        pP = AfP @ transpose(A) + Q

        C = transpose(solve(pP, AfP))

        sm = fm + C @ (sm - pm)
        sP = fP + C @ (sP - pP) @ transpose(C)
        if return_full:
            return (sm, sP), (sm.reshape(-1, 1), build_block_diag(sP), build_block_diag(C))
        else:
            return (sm, sP), (H @ sm.reshape(-1, 1), H @ build_block_diag(sP) @ transpose(H), build_block_diag(C))

    _, (sms, sPs, gains) = scan(f=body,
                                init=(fms[-1], fPs[-1]),
                                xs=(fms, fPs, As, Qs),
                                reverse=True)
    return sms, sPs, gains


def _parallel_rts_mf(fms, fPs, As, Qs, H, return_full, block_index):

    @vmap
    def build_block_diag(P_blocks):
        P = Pzeros.at[block_index].add(P_blocks.flatten())
        return P

    def build_mean(m):
        return m.reshape(num_time_steps, state_dim, 1)

    num_time_steps = As.shape[0]
    num_latents = As.shape[1]
    sub_state_dim = As.shape[2]
    state_dim = num_latents * sub_state_dim
    Pzeros = np.zeros([state_dim, state_dim])

    # build the associative smoothing elements
    smoothing_elems = parallel_smoothing_element(As, Qs, fms, fPs)
    gains = build_block_diag(smoothing_elems[0])
    last_elems = last_parallel_smoothing_element(fms[-1], fPs[-1])
    initial_elements = tuple(np.append(gen_es[:-1], np.expand_dims(last_e, 0), axis=0)
                             for gen_es, last_e in zip(smoothing_elems, last_elems))

    # run the parallel smoother
    final_elements = associative_scan(parallel_smoothing_operator, initial_elements, reverse=True)  # note vmap
    sms, sPs = build_mean(final_elements[1]), build_block_diag(final_elements[2])

    if return_full:
        return sms, sPs, gains
    else:
        return H @ sms, H @ sPs @ H.T, gains


def rauch_tung_striebel_smoother_meanfield(dt, kernel, filter_mean, filter_cov, return_full=False,
                                           parallel=False, block_index=None):
    """
    Run the RTS smoother to get p(fₙ|y₁,...,y_N) with a mean-field approximation for the latent components.
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param filter_mean: the intermediate distribution means computed during filtering [N, state_dim, 1]
    :param filter_cov: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
    :param return_full: a flag determining whether to return the full state distribution or just the function(s)
    :param parallel: flag to switch between parallel and sequential implementation of smoother
    :param block_index: mean-field block indiced required to build the block-diagonal model matrices from the blocks
    :return:
        smoothed_mean: the posterior marginal means [N, obs_dim]
        smoothed_var: the posterior marginal variances [N, obs_dim]
    """
    Pinf = kernel.stationary_covariance_meanfield()

    As = vmap(kernel.state_transition_meanfield)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    H = kernel.measurement_model()

    if parallel:
        means, covs, gains = _parallel_rts_mf(filter_mean, filter_cov, As, Qs, H, return_full, block_index)
    else:
        means, covs, gains = _sequential_rts_mf(filter_mean, filter_cov, As, Qs, H, return_full, block_index)
    return means, covs, gains


def _sequential_kf_pairs_mf(As, Qs, ys, noise_covs, m0, P0, block_index):

    def build_block_diag(P_blocks):
        P = np.zeros([state_dim, state_dim])
        P = P.at[block_index].add(P_blocks.flatten())
        return P

    def get_block_mean(m):
        return m.reshape(num_latents, sub_state_dim, 1)

    def get_block_cov(P):
        return P[block_index].reshape(num_latents, sub_state_dim, sub_state_dim)

    num_latents = m0.shape[0]
    sub_state_dim = m0.shape[1]
    state_dim = num_latents * sub_state_dim

    def body(carry, inputs):
        y, A, Q, obs_cov = inputs
        m_left, P_left, ell = carry

        # predict
        m_right = (A @ m_left).reshape(-1, 1)
        P_right = build_block_diag(A @ P_left @ transpose(A) + Q)

        # construct the joint distribution p(uₙ₋₁,uₙ) = p(uₙ₋₁)p(uₙ|uₙ₋₁)
        PA_ = build_block_diag(P_left @ transpose(A))
        m_joint = np.block([[m_left.reshape(-1, 1)],
                            [m_right]])
        P_joint = np.block([[build_block_diag(P_left), PA_],
                            [PA_.T, P_right]])

        S = P_joint + obs_cov

        ell_n = mvn_logpdf(y, m_joint, S)
        ell = ell + ell_n

        K = solve(S, P_joint).T

        # perform update
        m = m_joint + K @ (y - m_joint)
        P = P_joint - P_joint @ K.T

        # marginalise and store the now fully updated left state, uₙ₋₁
        m_left = get_block_mean(m[:state_dim])
        P_left = get_block_cov(P[:state_dim, :state_dim])
        # marginalise and propagate the right state, uₙ
        m_right = get_block_mean(m[state_dim:])
        P_right = get_block_cov(P[state_dim:, state_dim:])

        return (m_right, P_right, ell), (m_left, P_left)

    (_, _, loglik), (fms, fPs) = scan(f=body,
                                      init=(m0, P0, 0.),
                                      xs=(ys, As, Qs, noise_covs))
    return loglik, fms[1:], fPs[1:]  # discard intial dummy state


def kalman_filter_pairs_meanfield(dt, kernel, y, noise_cov, parallel=False, block_index=None):
    """
    A Kalman filter over pairs of states, with a mean-field approximation for the latent components.
    Assumes a heteroscedastic Gaussian observation model, i.e. var is vector valued
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, 2state_dim, 1]
    :param noise_cov: observation noise covariances [N, 2state_dim, 2state_dim]
    :param parallel: flag to switch between parallel and sequential implementation of Kalman filter
    :param block_index: mean-field block indices required to build the block-diagonal model matrices from the blocks
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """

    Pinf = kernel.stationary_covariance_meanfield()
    minf = np.zeros([Pinf.shape[0], Pinf.shape[1], 1])

    As = vmap(kernel.state_transition_meanfield)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)

    if parallel:
        raise NotImplementedError("Parallel KF not implemented yet")
    else:
        ell, means, covs = _sequential_kf_pairs_mf(As, Qs, y, noise_cov, minf, Pinf, block_index)
    return ell, (means, covs)


def dare(A, H, Q, R, Pinit, num_iters=20):
    """
    solve a discrete algebraic Ricatti equation
    :param A: transition matrix
    :param H: measurement model mean
    :param Q: process noise covariance
    :param R: observation noise covariance
    :param Pinit: initialisation
    :param num_iters: number of iterations
    :return:
    """

    # def body(X, _):
    #     K = A @ transpose(solve(H @ X @ transpose(H) + R, H @ X))
    #     X = (A - K @ H) @ X @ transpose(A - K @ H) + K @ R @ transpose(K) + Q
    #     return X, X

    def body(X, _):  # slightly faster version
        HX = H @ X
        S = HX @ H.T + R
        K = solve(S, HX).T
        X = A @ (X - K @ HX) @ A.T + Q
        return X, X

    X_final, _ = scan(f=body,
                      init=Pinit,
                      xs=np.zeros(num_iters))

    return X_final


def _sequential_kf_ih(AKHAs, HA, Kys, Ss, ys, m0, masks):

    def body(carry, inputs):
        y, AKHA, Ky, S, mask = inputs
        m, ell = carry

        obs_mean = HA @ m

        ell_n = mvn_logpdf(y, obs_mean, S, mask)
        ell = ell + ell_n

        # perform update
        m = AKHA @ m + Ky

        return (m, ell), m

    (_, loglik), (fms) = scan(f=body,
                              init=(m0, 0.),
                              xs=(ys, AKHAs, Kys, Ss, masks))
    return loglik, fms


@vmap
def parallel_filtering_operator_ih_homoscedastic(elem1, elem2):
    B1, v1 = elem1
    B2, v2 = elem2
    return B1 @ B2, B2 @ v1 + v2


@vmap
def parallel_filtering_operator_ih_heteroscedastic(elem1, elem2):
    B1, v1 = elem1
    B2, v2 = elem2
    return B1 @ B2, jsc_solve(B2, v1, sym_pos=False) + v2


def _parallel_kf_ih(AKHAs_or_iAKHAs, HA, Kys, Ss, ys, m0, masks, heteroscedastic):

    # perform parallel filtering
    if heteroscedastic:
        # Kys = index_add(Kys, index[0], AKHAs[1] @ m0)  # incorporate m0, TODO: not sure if correct, check
        final_elements = associative_scan(parallel_filtering_operator_ih_heteroscedastic, (AKHAs_or_iAKHAs, Kys))
    else:
        # Kys = index_add(Kys, index[0], jsc_inv(AKHAs[1]) @ m0)  # incorporate m0, TODO: not sure if correct, check
        final_elements = associative_scan(parallel_filtering_operator_ih_homoscedastic, (AKHAs_or_iAKHAs, Kys))
    fms = final_elements[1]

    # now compute the log likelihood
    obs_means = HA @ np.concatenate([m0[None], fms[:-1]])
    loglik = np.sum(vmap_mvn_logpdf(ys, obs_means, Ss, masks))

    return loglik, fms


def kalman_filter_infinite_horizon(dt, kernel, y, noise_cov, mask=None, parallel=False,
                                   heteroscedastic=False, noise_cov_tied=None, dare_iters=20, dare_init=None):
    """
    Run the Kalman filter to get p(fₙ|y₁,...,yₙ).
    Assumes a heteroscedastic Gaussian observation model, i.e. var is vector valued
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, D, 1]
    :param noise_cov: observation noise covariances [N, D, D]
    :param mask: boolean mask for the observations (to indicate missing data locations) [N, D, 1]
    :param parallel: flag to switch between parallel and sequential implementation of Kalman filter
    :param heteroscedastic: flag to indicate whether the approximate model has heteroscedastic noise
    :param noise_cov_tied: averaged observation noise covariance [N, D, D]
    :param dare_iters: number of iterations to run the DARE solver for
    :param dare_init: initialisation for the DARE solver
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """
    if mask is None:
        mask = np.zeros_like(y, dtype=bool)
    Pinf = kernel.stationary_covariance()
    minf = np.zeros([Pinf.shape[0], 1])
    A = kernel.state_transition(dt[1])
    Q = process_noise_covariance(A, Pinf)
    H = kernel.measurement_model()

    # solve the discrete algebraic Ricatti equation to compute the steady-state covariance
    dare_init = Pinf if dare_init is None else dare_init
    Pdare = dare(A, H, Q, noise_cov_tied, dare_init, dare_iters)

    # innovation variance
    S = H @ Pdare @ H.T + noise_cov_tied
    # stationary gain
    K = Pdare @ solve(S, H).T
    # precalculate
    HA = H @ A

    N = dt.shape[0]
    if heteroscedastic:
        # these operations are O(Ns^3) (s=num spatial pts), but can be run in parallel
        Ss = H @ Pdare @ H.T + noise_cov
        Ks = Pdare @ transpose(solve(Ss, np.tile(H, [N, 1, 1])))  # <-- bottleneck
        if not parallel:
            AKHAs = A - Ks @ HA
    else:
        # the homoscedastic version (Gaussian likelihood & no missing data) is O(Nd^2)
        Ss = np.tile(S, [N, 1, 1])
        Ks = np.tile(K, [N, 1, 1])
        AKHAs = np.tile(A - K @ HA, [N, 1, 1])

    # precalculate
    Kys = Ks @ y

    # infinite-horizon state covariance
    cov = Pdare - K @ H @ Pdare

    if parallel:
        if heteroscedastic:
            iA = kernel.state_transition(-dt[1])  # inv(A)
            iAKHAs = iA + iA @ ((Pdare @ H.T) @ solve(noise_cov, np.tile(H, [N, 1, 1])))  # inv(AKHAs)
            ell, means = _parallel_kf_ih(iAKHAs, HA, Kys, Ss, y, minf, mask, heteroscedastic)
        else:
            # AKHAs = A - Ks @ HA
            ell, means = _parallel_kf_ih(AKHAs, HA, Kys, Ss, y, minf, mask, heteroscedastic)
    else:
        ell, means = _sequential_kf_ih(AKHAs, HA, Kys, Ss, y, minf, mask)

    covs = (Pdare, cov)

    return ell, (means, covs)


def rts_dare(A, Q, Pinf, num_iters=20):
    """
    solve a discrete algebraic Ricatti equation
    :param A: transition matrix
    :param Q: process noise covariance
    :param Pinf: initialisation
    :param num_iters: number of iterations
    :return:
    """

    def body(X, _):
        X = A @ X @ A.T + Q
        return X, X

    X_final, _ = scan(f=body,
                      init=Pinf,
                      xs=np.zeros(num_iters))

    return X_final


def _sequential_rts_ih(fms, Afms, H, gain, return_full):

    def body(sm, inputs):
        fm, Afm = inputs

        sm = fm + gain @ (sm - Afm)

        if return_full:
            return sm, sm
        else:
            return sm, H @ sm

    _, sms = scan(f=body,
                  init=fms[-1],
                  xs=(fms, Afms),
                  reverse=True)
    return sms


@vmap
def parallel_smoothing_operator_ih(elem1, elem2):
    B1, v1 = elem1
    B2, v2 = elem2
    return B1 @ B2, B2 @ v1 + v2


def _parallel_rts_ih(fms, Afms, H, gain, return_full):

    gains = np.tile(gain, [fms.shape[0], 1, 1])
    smoothing_elems = (gains, fms - gains @ Afms)
    last_elems = (np.zeros_like(gain), fms[-1])
    initial_elements = tuple(np.append(gen_el, np.expand_dims(last_el, 0), axis=0)
                             for gen_el, last_el in zip(smoothing_elems, last_elems))
    final_elements = associative_scan(parallel_smoothing_operator_ih, initial_elements, reverse=True)
    sms = final_elements[1][:-1]

    if return_full:
        return sms
    else:
        return H @ sms


def rauch_tung_striebel_smoother_infinite_horizon(dt, kernel, filter_mean, filter_cov, return_full=False,
                                                  parallel=False, dare_iters=20, dare_init=None):
    """
    Run the RTS smoother to get p(fₙ|y₁,...,y_N),
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param filter_mean: the intermediate distribution means computed during filtering [N, state_dim, 1]
    :param filter_cov: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
    :param return_full: a flag determining whether to return the full state distribution or just the function(s)
    :param parallel: flag to switch between parallel and sequential implementation of smoother
    :param dare_iters: number of iterations to run the DARE solver for
    :param dare_init: initialisation for the DARE solver
    :return:
        smoothed_mean: the posterior marginal means [N, obs_dim]
        smoothed_var: the posterior marginal variances [N, obs_dim]
    """
    Pinf = kernel.stationary_covariance()
    A = kernel.state_transition(dt[0])
    H = kernel.measurement_model()
    N = dt.shape[0]

    (Pdare, filter_cov) = filter_cov  # unpack filter result
    state_dim = A.shape[1]
    if Pdare.shape[0] != state_dim:  # the sparse Markov case
        Pdare = Pdare[state_dim:, state_dim:]

    # the stationary gain
    gain = filter_cov @ solve(Pdare, A).T

    # solve the Riccati equation
    Qdare = filter_cov - gain @ Pdare @ gain.T
    # Qdare = (Qdare + transpose(Qdare)) / 2
    dare_init = Pinf if dare_init is None else dare_init
    dare_cov = rts_dare(gain, Qdare, dare_init, dare_iters)

    # pre-compute
    Afms = A @ filter_mean

    if parallel:
        means = _parallel_rts_ih(filter_mean, Afms, H, gain, return_full)
    else:
        means = _sequential_rts_ih(filter_mean, Afms, H, gain, return_full)

    if return_full:
        cov = dare_cov
    else:
        cov = H @ dare_cov @ H.T
    covs = np.tile(cov, [N, 1, 1])
    gains = np.tile(gain, [N, 1, 1])

    return means, covs, gains, dare_cov


def dare_pairs(A, Q, R, Pinit, num_iters=20):
    """
    solve a discrete algebraic Ricatti equation
    :param A: transition matrix
    :param Q: process noise covariance
    :param R: observation noise covariance
    :param Pinit: initialisation
    :param num_iters: number of iterations
    :return:
    """
    state_dim = A.shape[0]

    def body(X, _):
        S = X + R
        K = solve(S, X).T
        X = X - K @ X  # update
        X_left = X[state_dim:, state_dim:]
        X_right = A @ X_left @ A.T + Q  # predict
        XA_ = X_left @ A.T
        X = np.block([[X_left, XA_],
                      [XA_.T,  X_right]])
        return X, X

    X_final, _ = scan(f=body,
                      init=Pinit,
                      xs=np.zeros(num_iters))

    return X_final


def _sequential_kf_ih_pairs(IKs, A, Kys, Ss, ys, m0):
    state_dim = A.shape[0]

    def body(carry, inputs):
        y, IK, Ky, S = inputs
        m_left, ell = carry

        # predict
        m_right = A @ m_left

        # construct the joint distribution p(uₙ₋₁,uₙ) = p(uₙ₋₁)p(uₙ|uₙ₋₁)
        m_joint = np.block([[m_left],
                            [m_right]])

        ell_n = mvn_logpdf(y, m_joint, S)
        ell = ell + ell_n

        # perform update
        m = IK @ m_joint + Ky

        # marginalise and store the now fully updated left state, uₙ₋₁
        m_left = m[:state_dim]
        # marginalise and propagate the right state, uₙ
        m_right = m[state_dim:]

        return (m_right, ell), m_left

    (_, loglik), fms = scan(f=body,
                            init=(m0, 0.),
                            xs=(ys, IKs, Kys, Ss))
    return loglik, fms[1:]  # discard intial dummy state


def kalman_filter_infinite_horizon_pairs(dt, kernel, y, noise_cov, parallel=False,
                                         noise_cov_tied=None, dare_iters=20, dare_init=None):
    """
    A Kalman filter over pairs of states, in which y is [2D, 1] and noise_cov is [2D, 2D]
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, D, 1]
    :param noise_cov: observation noise covariances [N, D, D]
    :param parallel: flag to switch between parallel and sequential implementation of Kalman filter
    :param noise_cov_tied: averaged observation noise covariance [N, D, D]
    :param dare_iters: number of iterations to run the DARE solver for
    :param dare_init: initialisation for the DARE solver
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """

    Pinf = kernel.stationary_covariance()
    minf = np.zeros([Pinf.shape[0], 1])

    A = kernel.state_transition(dt[1])
    Q = process_noise_covariance(A, Pinf)

    # solve the discrete algebraic Ricatti equation to compute the steady-state covariance
    dare_init = Pinf if dare_init is None else dare_init
    Pdare = dare_pairs(A, Q, noise_cov_tied, dare_init, dare_iters)

    # innovation variance
    S = Pdare + noise_cov_tied
    # stationary gain
    K = solve(S, Pdare).T

    # these operations are O(Nd^3), but can be run in parallel, so the full algorith would be O(Nd^2) on a GPU
    Ss = Pdare + noise_cov
    Ks = transpose(vmap(solve, [0, None])(Ss, Pdare))  # <-- bottleneck
    IKs = np.eye(K.shape[0]) - Ks

    # precalculate
    Kys = Ks @ y

    # infinite-horizon state covariance
    cov = Pdare - K @ Pdare

    if parallel:
        raise NotImplementedError("Parallel KF not implemented yet")
    else:
        ell, means = _sequential_kf_ih_pairs(IKs, A, Kys, Ss, y, minf)

    state_dim = A.shape[0]
    covs = (Pdare, cov[:state_dim, :state_dim])

    return ell, (means, covs)
