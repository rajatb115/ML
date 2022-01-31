import numpy as np
from scipy import stats

debug = False


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
            rho = float(config['config']['alpha']) * stats.multivariate_normal.pdf(pixel,
                                                                                   mean[gaussian * 3: gaussian * 3 + 3],
                                                                                   var[gaussian] * np.eye(3))
            if debug:
                print("rho :", rho)

            mean[gaussian * 3: gaussian * 3 + 3] = (1 - rho) * mean[gaussian * 3: gaussian * 3 + 3] + rho * pixel
            var[gaussian] = (1 - rho) * var[gaussian] + rho * np.dot(distance, distance)

            # If matched then weight = (1-alpha)*weight + alpha*M (M=0 if does not match, M=1 if matched)
            weight[gaussian] = (1 - float(config['config']['alpha'])) * weight[gaussian] + float(
                config['config']['alpha'])

            # exit if matched
            break
        else:
            weight[gaussian] = (1 - float(config['config']['alpha'])) * weight[gaussian]

    # If none of the gaussian matched then check if you want to add a new gaussian or replace an existing gaussian
    update_gaussian_index = 0
    if not matched:
        if gaussian_cnt < int(config['config']['number_gaussian']):
            # add a new gaussian
            gaussian_cnt += 1
            update_gaussian_index = gaussian_cnt - 1
        else:
            # replace an existing gaussian
            # index of the gaussian to be updated is the gaussian with the minimum weight/variance ratio
            update_gaussian_index = 0
            ratio = weight[update_gaussian_index] / var[update_gaussian_index]
            for i in range(1, gaussian_cnt):
                if weight[i] / var[i] < ratio:
                    update_gaussian_index = i
                    ratio = weight[i] / var[i]

        # Setting the value of new gaussian
        # new mean = value of the pixel
        mean[update_gaussian_index * 3:update_gaussian_index * 3 + 3] = pixel
        var[update_gaussian_index] = config['config']['variance']
        weight[update_gaussian_index] = config['config']['weight']

        if debug:
            print(mean)
            print(var)
            print(weight)

    # Normalize the weights
    sm_weight = np.sum(weight)
    weight = np.divide(weight, sm_weight)

    if debug:
        print(weight)
        print(gaussian_cnt)

    # segment the foreground and background (background <= 6.25 sig)
    # sort the weight/var values
    sorted_values = np.argsort(np.divide(weight[0:gaussian_cnt], var[0:gaussian_cnt]))
    sum_weight = 0

    return pixel, mean, var, weight
