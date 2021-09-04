import numpy as np
from numpy import savetxt
from scipy.misc import imread, imsave, imresize
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from skimage import color
import global_variables
import cv2
import matplotlib.pyplot as plt
def get_descriptors (image1, image2, window_size, pca_dim_gray, pca_dim_rgb):

    #################################################   grayscale-diff (abs)

    descriptors = np.zeros((image1.shape[0],image1.shape[1], window_size * window_size))
    diff_image = cv2.absdiff(image1, image2)
    diff_image = color.rgb2gray(diff_image)
    imsave(global_variables.output_dir  + '/diff.jpg', diff_image)
    diff_image = np.pad(diff_image,((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
            'constant')  # default is 0
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            descriptors[i,j,:] =diff_image[i:i+window_size,j:j+window_size].ravel()
    descriptors_gray_diff = descriptors.reshape((descriptors.shape[0] * descriptors.shape[1], descriptors.shape[2]))

    #################################################   3-channels-diff (abs)

    descriptors = np.zeros((image1.shape[0], image1.shape[1], window_size * window_size*3))
    diff_image_r = cv2.absdiff(image1[:, :, 0],image2[:, :, 0])
    diff_image_g = cv2.absdiff(image1[:, :, 1],image2[:, :, 1])
    diff_image_b = cv2.absdiff(image1[:, :, 2],image2[:, :, 2])

    if (global_variables.save_extra_stuff):
        imsave(global_variables.output_dir + '/final_diff.jpg', cv2.absdiff(image1, image2))
        imsave(global_variables.output_dir +'/final_diff_r.jpg', diff_image_r)
        imsave(global_variables.output_dir + '/final_diff_g.jpg', diff_image_g)
        imsave(global_variables.output_dir + '/final_diff_b.jpg', diff_image_b)

    diff_image_r = np.pad(diff_image_r, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                        'constant')  # default is 0
    diff_image_g = np.pad(diff_image_g,
                            ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                            'constant')  # default is 0
    diff_image_b = np.pad(diff_image_b,
                            ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                            'constant')  # default is 0

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            feature_r = diff_image_r[i:i + window_size, j:j + window_size].ravel()
            feature_g = diff_image_g[i:i + window_size, j:j + window_size].ravel()
            feature_b = diff_image_b[i:i + window_size, j:j + window_size].ravel()
            descriptors[i, j, :] = np.concatenate((feature_r, feature_g, feature_b))
    descriptors_rgb_diff = descriptors.reshape((descriptors.shape[0] * descriptors.shape[1], descriptors.shape[2]))

    #################################################   concatination

    descriptors_gray_diff = descriptors_to_pca(descriptors_gray_diff, pca_dim_gray,window_size)
    descriptors_colored_diff = descriptors_to_pca(descriptors_rgb_diff, pca_dim_rgb,window_size)

    descriptors = np.concatenate((descriptors_gray_diff, descriptors_colored_diff), axis=1)

    return descriptors


#assumes descriptors is already flattened
#returns descriptors after moving them into the PCA vector space
def descriptors_to_pca(descriptors, pca_target_dim, window_size):
    vector_set, mean_vec = find_vector_set(descriptors,window_size)
    pca = PCA(pca_target_dim)
    pca.fit(vector_set)
    EVS = pca.components_
    mean_vec = np.dot(mean_vec, EVS.transpose())
    FVS = find_FVS(descriptors, EVS.transpose(), mean_vec)
    return FVS


#The returned vector_set goes later to the PCA algorithm which derives the EVS (Eigen Vector Space).
#Therefore, there is a mean normalization of the data
#jump_size is for iterating non-overlapping windows. This parameter should be eqaul to the window_size of the system
def find_vector_set(descriptors, jump_size):
    descriptors_2d = descriptors.reshape((global_variables.size_0, global_variables.size_1, descriptors.shape[1]))
    vector_set = descriptors_2d[::jump_size,::jump_size]
    vector_set = vector_set.reshape((vector_set.shape[0]*vector_set.shape[1], vector_set.shape[2]))
    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec  # mean normalization
    return vector_set, mean_vec

#returns the FSV (Feature Vector Space) which then goes directly to clustering (with Kmeans)
#Multiply the data with the EVS to get the entire data in the PCA target space
def find_FVS(descriptors, EVS, mean_vec):
    FVS = np.dot(descriptors, EVS)
    FVS = FVS - mean_vec
    print("\nfeature vector space size", FVS.shape)
    return FVS

#Creates the change map, according to a specified number of clusters and the other parameters
def compute_change_map(image1, image2, window_size=5, clusters=16, pca_dim_gray=3, pca_dim_rgb=9):
    descriptors = get_descriptors(image1, image2, window_size, pca_dim_gray, pca_dim_rgb)
    # Now we are ready for clustering!
    change_map = Kmeansclustering(descriptors, clusters, image1.shape)
    mse_array, size_array = clustering_to_mse_values(change_map, image1, image2, clusters)
    sorted_indexes = np.argsort(mse_array)
    colors_array = [plt.cm.jet(float(np.argwhere(sorted_indexes == class_))/(clusters-1)) for class_ in range(clusters)]
    colored_change_map = np.zeros((change_map.shape[0], change_map.shape[1], 3), np.uint8)
    for i in range(change_map.shape[0]):
        for j in range(change_map.shape[1]):
            colored_change_map[i, j]= (255*colors_array[change_map[i,j]][0],255*colors_array[change_map[i,j]][1],255*colors_array[change_map[i,j]][2])

    if (global_variables.save_extra_stuff):
        imsave(global_variables.output_dir+ '/window_size_'+str(window_size)+'_pca_dim_gray'+str(pca_dim_gray)+'_pca_dim_rgb'
               +str(pca_dim_rgb)+'_clusters_'+  str(clusters) + '.jpg', colored_change_map)

    #Saving Output for later evaluation
    savetxt(global_variables.output_dir+ '/clustering_data.csv', change_map, delimiter=',')
    return change_map, mse_array, size_array

def Kmeansclustering(FVS, components, images_size):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    flatten_change_map = kmeans.predict(FVS)
    change_map = np.reshape(flatten_change_map, (images_size[0],images_size[1]))
    return change_map

#calculates the mse value for each cluster of change_map
def clustering_to_mse_values(change_map, img1, img2, n):
    mse =  [0.0 for i in range (0,n)]
    size = [0 for i in range (0,n)]
    img1 = img1.astype(int)
    img2 = img2.astype(int)
    for i in range(change_map.shape[0]):
        for j in range(change_map.shape[1]):
            mse[change_map[i,j]] += np.mean((img1[i,j]-img2[i,j])**2)
            size[change_map[i,j]] += 1
    return [(mse[k]/(255**2))/size[k] for k in range (0,n)], size


#clustering is the clustering map which is de-facto a list that in index #class_ has a list of indexes of the pixels that belong to that class.
#n is the number of classes
#img1 and img2 are the 2 images to compute the MSE values on
#the function returns a list of arrays, in each there are the classes that should be for this result. Each array is called a combination.
def find_groups(MSE_array, size_array, n, problem_size):
    results_groups = []
    class_number_arr = [x for x in range(n)]
    plt.figure()
    plt.xticks(class_number_arr)
    plt.xlabel('Index')
    plt.ylabel('MSE')
    zipped = zip(MSE_array,size_array, class_number_arr)
    #sort according to increasing MSE values
    zipped= sorted(zipped)
    max_mse = np.max(MSE_array)
    zipped_filtered = [(mse, size, class_num) for mse, size, class_num in zipped if (mse>= 0.1 * max_mse and size<0.1*problem_size)]
    MSE_filtered_sorted = [mse for mse, size, class_num in zipped_filtered]
    number_class_filtered_sorted = [class_num for mse, size, class_num in zipped_filtered]

    #save output for later evaluation if needed
    savetxt(global_variables.output_dir + '/mse_filtered_sorted.csv', MSE_filtered_sorted, delimiter=',')
    savetxt(global_variables.output_dir + '/classes_filtered_sorted.csv', number_class_filtered_sorted, delimiter=',')

    print(MSE_filtered_sorted[::-1]) #decreasing MSE values
    plt.scatter([i for i in range(len(MSE_filtered_sorted))], MSE_filtered_sorted[::-1] , c='red')
    plt.savefig(global_variables.output_dir+"/mse.png")

    consecutive_diff = np.diff(MSE_filtered_sorted)
    if len(number_class_filtered_sorted) == 0:
        print("No (small) changes detected")
        exit(0)
    elif len(consecutive_diff) ==0:
        results_groups.append([number_class_filtered_sorted[0]])
    else:
        max = len(number_class_filtered_sorted)-1
        while (max >0 and num_results>0):
            num_results = num_results -1
            max = np.argmax(consecutive_diff)
            consecutive_diff = consecutive_diff[:max]
            results_groups.append(number_class_filtered_sorted[max+1:])
        if(max==0 and num_results>0):
            results_groups.append(number_class_filtered_sorted)
    return results_groups

#selects the classes to be shown to the user as 'changes'.
#this selection is done by an MSE heuristic using DBSCAN clustering, to seperate the highest mse-valued classes from the others.
#the eps density parameter of DBSCAN might differ from system to system
def find_group_of_accepted_classes_DBSCAN(MSE_array):
    print(MSE_array)
    clustering = DBSCAN(eps=0.02, min_samples=1).fit(np.array(MSE_array).reshape(-1,1))
    number_of_clusters = len(set(clustering.labels_))
    print(clustering.labels_)
    classes = [[] for i in range(number_of_clusters)]
    centers = [0 for i in range(number_of_clusters)]
    for i in range(len(MSE_array)):
        centers[clustering.labels_[i]] += MSE_array[i]
        classes[clustering.labels_[i]].append(i)

    centers = [centers[i]/len(classes[i]) for i in range(number_of_clusters)]
    min_class = centers.index(min(centers))
    accepted_classes = []
    for i in range(len(MSE_array)):
        if clustering.labels_[i] != min_class:
            accepted_classes.append(i)
    plt.figure()
    plt.xlabel('Index')
    plt.ylabel('MSE')
    plt.scatter(range(len(MSE_array)), MSE_array, c="red")
    print(accepted_classes)
    print(np.array(MSE_array)[np.array(accepted_classes)])
    plt.scatter(accepted_classes[:], np.array(MSE_array)[np.array(accepted_classes)], c="blue")
    plt.title('K Mean Classification')
    plt.savefig(global_variables.output_dir+"/mse.png")

    #save output for later evaluation
    savetxt(global_variables.output_dir + '/accepted_classes.csv', accepted_classes, delimiter=',')
    return [accepted_classes]

    #save output for later evaluation
    savetxt(global_variables.output_dir + '/accepted_classes.csv', accepted_classes, delimiter=',')
    return [accepted_classes]

#the 'changes' are drawn on the input image (with some transparency)
#combination is the list of classes to appear in the result.
#the color of each class is determined by its order of magnitude according to a jet palette.
def draw_combination_on_transparent_input_image(classes_mse, clustering, combination, transparent_input_image):

    # HEAT MAP ACCORDING TO MSE ORDER
    sorted_indexes = np.argsort(classes_mse)
    for class_ in combination:
        c = plt.cm.jet(float(np.argwhere(sorted_indexes == class_))/(len(classes_mse)-1))
        for [i, j] in clustering[class_]:
            transparent_input_image[i, j] = (c[2] * 255, c[1] * 255, c[0] * 255, 255)  #BGR
    return transparent_input_image

