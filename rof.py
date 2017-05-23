import numpy as np


def denoise(im, U_init, tolerance=0, tau=.125, tv_weight=100):
    """An implementation of the Rudin-Osher-Fatemi (ROF)
       denoissing model using the numerical procedure presented in
       eq (11) A. Chambolle (2005)

       Inputs:
             im: noisy input image (grayscale)
             U_init: initial guess for U
             tolerance: tolerance for stop criterion
             tau: steplength
             tv_weight: TV-regularization term
       Returns:
             Denoised & detextured image, texture residual
    """
    # Size of noisy image
    m, n = im.shape

    # Initialize

    U = U_init
    # x-component of the dual field
    Px = im
    # y component of the dual field
    Py = im
    error = 1

    while error > tolerance:
        Uold = U

        # Gradient of primal variable

        # x-component of of U's gradient
        GradUx = np.roll(U, -1, axis=1) - U

        # y-component of U's gradient
        GradUy = np.roll(U, -1, axis=0) - U

        # Update the dual variable
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy
        NormNew = np.maximum(1, np.sqrt(PxNew**2 + PyNew**2))

        # Update x and y components
        Px = PxNew / NormNew
        Py = PyNew / NormNew

        # Update the primal variable

        # Right translation of the x and y components
        RxPx = np.roll(Px, 1, axis=1)
        RyPy = np.roll(Py, 1, axis=0)

        # Calculate divergence of the dual field
        DivP = (Px - RxPx) + (Py - RyPy)

        # Update primal variable
        U = im + tv_weight * DivP

        # Update error

        error = np.linalg.norm(U - Uold) / np.sqrt(n * m)

        # Denoised image and texture residual
        return U, im - U
