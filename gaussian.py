import numpy as np
from scipy import stats

debug = True


def process_pixel(cstm_args):
    pixel = cstm_args[0]
    mean = cstm_args[1]
    weight = cstm_args[2]
    var = cstm_args[3]
    gaussian_cnt = int(cstm_args[4])
    config = cstm_args[5]

    # variable to check if value of pixel is matching any gaussian
    matched = False

    if debug:
        print("# At gaussian values are :")
        print("Mean :", mean)
        print("Variance :", var)
        print("Pixel :", pixel)
        print("Weight :", weight)
        print("Gaussian Count :", gaussian_cnt)

    # check for the matching gaussian
    for gaussian in range(gaussian_cnt):
        # check if matches with one of the gaussian ~2.5 sig (6.25 var)
        # find distance of the pixel from the mean
        distance = pixel - mean[gaussian * 3: gaussian * 3 + 3]

        if debug:
            print("Distance :", distance)

        if np.dot(distance, distance) <= 6.25 * var[gaussian]:
            matched = True

            # update the parameters
            #######################
            rho = float(config['config']['alpha']) * stats.multivariate_normal.pdf(pixel, mean[gaussian * 3: gaussian * 3 + 3], var[gaussian]*np.eye(3))
            if debug:
                print("rho :", rho)

            mean[gaussian*3: gaussian*3+3] = (1-rho)*mean[gaussian*3: gaussian*3+3] + rho * pixel
            var[gaussian] = (1-rho)*var[gaussian] + rho * np.dot(distance, distance)

            # If matched then weight = (1-alpha)*weight + alpha*M (M=0 if does not match, M=1 if matched)
            weight[gaussian] = (1-float(config['config']['alpha']))*weight[gaussian] + float(config['config']['alpha'])

            # exit if matched
            break
        else:
            weight[gaussian] = (1 - float(config['config']['alpha'])) * weight[gaussian]

    # If none of the gaussian matched then check if you want to add a new gaussian or replace an existing gaussian
    if gaussian_cnt < int(config['config']['number_gaussian']):
        # add a new gaussian

    else:
        # replace an existing gaussian
