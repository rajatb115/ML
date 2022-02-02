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
    frame_history = cstm_args[6]
    frame_history_cnt = cstm_args[7]

    # variable to check if value of pixel is matching any gaussian
    matched = False

    if debug:
        log_info = open("data/log.txt", 'a')
        log_info.write("# At gaussian values are :\n")
        log_info.write("Mean : " + str(mean) + "\n")
        log_info.write("Variance : " + str(var) + "\n")
        log_info.write("Pixel : " + str(pixel) + "\n")
        log_info.write("Weight : " + str(weight) + "\n")
        log_info.write("Gaussian Count : " + str(gaussian_cnt) + "\n")
        log_info.close()

    # check for the matching gaussian
    for gaussian in range(gaussian_cnt):
        # check if matches with one of the gaussian ~2.5 sig (6.25 var)
        # find distance of the pixel from the mean
        distance = pixel - mean[gaussian * 3: gaussian * 3 + 3]

        if debug:
            log_info = open("data/log.txt", 'a')
            log_info.write("Distance : " + str(distance)+"\n")
            log_info.close()

        if np.dot(distance, distance) <= 6.25 * var[gaussian]:
            matched = True
            # update the parameters
            rho = stats.multivariate_normal.pdf(pixel, mean[gaussian * 3: gaussian * 3 + 3], var[gaussian] * np.eye(3))
            rho = float(config['config']['alpha']) * rho

            if debug:
                log_info = open("data/log.txt", 'a')
                log_info.write("rho : " + str(rho)+"\n")
                log_info.close()

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
            log_info = open("data/log.txt", 'a')
            log_info.write("Values after the update of the gaussian \n")
            log_info.write("Mean : " + str(mean) + "\n")
            log_info.write("Variance : " + str(var) + "\n")
            log_info.write("Weight : " + str(weight) + "\n")
            log_info.write("\n")
            log_info.close()

    # Normalize the weights
    sm_weight = np.sum(weight)
    weight = np.divide(weight, sm_weight)

    if debug:
        log_info = open("data/log.txt", 'a')
        log_info.write("# At gaussian values are :\n")
        log_info.write("Values after normalizing the weights \n")
        log_info.write("Weight : " + str(weight) + "\n")
        log_info.write("Gaussian count : " + str(gaussian_cnt) + "\n")
        log_info.write("\n")
        log_info.close()

    # segment the foreground and background (background <= 6.25 sig)
    # sort the weight/var values
    sorted_values = np.argsort(np.divide(weight[0:gaussian_cnt], var[0:gaussian_cnt]))
    sum_weight = 0

    pixel_b = np.zeros(3)
    pixel_f = pixel
    t = False
    for i in sorted_values[::-1]:
        distance = pixel_f - mean[i * 3 : i * 3 + 3]
        if np.dot(distance, distance) <= 6.25 * var[i] and sum_weight < float(config['config']['weight_threshold']):
            pixel_f = 255 * np.ones(3)
            t = True

        sum_weight += weight[i]
        # if sum_weight > float(config['config']['weight_threshold']):
        #    break
    if not t:
        pixel_b[0] = np.mean(frame_history[:, 0])
        pixel_b[1] = np.mean(frame_history[:, 1])
        pixel_b[2] = np.mean(frame_history[:, 2])
        # print(pixel_b.shape)
    else:
        pixel_b = pixel
        # print(pixel_b.shape)

    # print(frame_history_cnt)

    if int(frame_history_cnt) < int(config['config']['history']):
        # print(frame_history.shape)
        frame_history[int(frame_history_cnt), 0] = pixel_b[0]
        frame_history[int(frame_history_cnt), 1] = pixel_b[1]
        frame_history[int(frame_history_cnt), 2] = pixel_b[2]
        frame_history_cnt += 1
    else:
        # frame_history = frame_history[1:]
        for kk in range(1, int(frame_history_cnt)):
            frame_history[kk - 1] = frame_history[kk]

        frame_history[len(frame_history) - 1, :] = pixel_b[:]

    return pixel_b, pixel_f, mean, var, weight, gaussian_cnt, frame_history, frame_history_cnt
