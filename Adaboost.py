#!/usr/bin/env python
# coding: utf-8
#
#
# Author :     Eoghan O'Connor
#
#
# File Name:    AdaBoost Classifier using weak linear classifiers
#
# Description:  A weak linear classifier is created.
#				This is implemented by creating an orientation vector
#				using the weighted mean of the positive and negative points
#				And finding the difference vector between them.
#				
#				The points are then projected onto the orientation vector
#				A seperation boundary , that seperates each pair of 
#				neighbouring points is created. This is perpendicular to 
#				the orientation vector. The most accurate seperation
#				boundary is choosen as the weak classifier.
#				
#				This weak classifier calculates its error based on how many
#				points it labels correctly. It then updates the points' weights
#				according to the error, i.e misclassified points are weighted more
#				heavily. These updated weights are used for training the next weak
#				classifier. A weight for the classifier itself is calculated also,
#				this is refered to as alpha.
#
#				The adaboost classifier combines all the weak linear classifiers
#				multiplied by their respective alpha values to classify points.
#
#				The classifications regions according to the adaboost classifier 
#				are plotted with the data set.		
#
#
#               
#                





# Function to calculate the weighted mean of a data set with an associated set of weights
def calc_mean_weighted(data_train, w):
    
    # Seperate training data into positive and negative sets according to label
    positives = data_train[np.where(data_train[:,2] == 1)]
    negatives = data_train[np.where(data_train[:,2] == -1)]
    
    # Create arrays of associated weights for postives and negative labeled data points
    pos_weights = w[np.where(data_train[:,2] == 1)]
    neg_weights = w[np.where(data_train[:,2] == -1)]
    
    # Apply weighted multiplication to respective points, positives and negative
    for i in range(len(positives)):
        positives[i][0]=positives[i][0]*pos_weights[i]
        positives[i][1]=positives[i][1]*pos_weights[i]
        
    for i in range(len(negatives)):
        negatives[i][0]=negatives[i][0]*neg_weights[i]
        negatives[i][1]=negatives[i][1]*neg_weights[i]
    
    # Find x,y mean points for positives and negative labeled points
    pos_x_mean = sum(positives)[0]/sum(pos_weights)
    pos_y_mean = sum(positives)[1]/sum(pos_weights)
    neg_x_mean = sum(negatives)[0]/sum(neg_weights)
    neg_y_mean = sum(negatives)[1]/sum(neg_weights)
    
    # Find slope of weighted mean points difference line. This is the orientation line for the deision boundary
    slope = float((neg_y_mean-pos_y_mean)/(neg_x_mean-pos_x_mean))
    constant = float(pos_y_mean - (slope*pos_x_mean))
    # return slope and constant for equation of line through weighted mean points
    return slope, constant


# Function to create an array of points that represent the midpoints between all
# training data point pairs projected onto the orientation line for a given weighted mean
def find_midpoints(data_train, m, b):
    # first we find the perpendicular slope to project the points
    m_inv = -1/m
  
    intersect_points = []
    
    # Calculate the point of intersections for vectors passing through each data point
    # that are normal to the orientation line calculated using the mean weighted points
    for i in range(len(data_train)):
        b2 = data_train[i][1] - (m_inv*data_train[i][0])
        A = np.array([[-m,1],[-m_inv,1]])
        B = np.array([b, b2])
        intersect_points.append(np.linalg.solve(A,B))

    # Sort all points of intersection according to x value (left to right)
    intersect_points = np.array(intersect_points)
    sorted_inter = intersect_points[np.argsort(intersect_points[:,0])]

    # Initialise array to store all midpoints with point outside all datapoints
    midpoints = []
    
    # Iterate through all pairs of points finding midpoints
    for i in range(len(sorted_inter)-1):
        x = (sorted_inter[i][0] + sorted_inter[i+1][0])/2
        y = (sorted_inter[i][1] + sorted_inter[i+1][1])/2
        b_local = y - (m_inv*x)
        midpoints.append([x,y,m_inv,b_local])
    return midpoints


# Function to determine the sign of the determinate of a point and a vector
def classify(data_point,point1, point2):
    # Generate two points that define a vector
    x1, y1 = point1[0], point1[1] 
    x2, y2 = point2[0], point2[1]
    
    # The point to be checked
    x, y = data_point[0], data_point[1]
    
    # return sign of determinate
    return np.sign(((x-x1)*(y2-y1))-((y-y1)*(x2-x1)))


# Function to create a weighted weak linear classifier and update the points' weights
def train(data_train, decision_points, w):
    # Initialise error as an arbitrarily high number
    error = 9999999
    
    # Loop through all decision boundary points and determine if points are misclassified or not
    for i in range(len(decision_points)):
        #Define two points for decision boundary
        point1,point2=decision_points[i][:2], data_train[i][:2]
        #initialise error to zero for each loop
        temp_error = 0
        # Iterate through all points and classify for current decision boundary
        for j in range(len(data_train)):
            # If point is misclassified, sum the associated weight for total weighted error
            if classify(data_train[j], point1, point2) != data_train[j][2]:
                temp_error += w[j] 
                
        # Update error with new minimum error if necessary and store best decision boundary
        if temp_error < error:
            error = temp_error
            gpoint1=point1
            gpoint2=point2
            
    # Print minimum error for each weak linear classifier
    #print(error)
    
    # Check if points are correctly classified for best decision boundary in current weak classifier
    # Update weights accordingly
    for i in range(len(w)):       
        if classify(data_train[i], gpoint1, gpoint2) != data_train[i][2]:
            w[i] *= 1/((2*error)+1e-10)
        else:
            w[i] *= 1/(2*(1-error))
                
    # Calculate alpha for current linear weak classifier
    alpha = 0.5*(np.log((1-error)/error))
    
    # Create weak learner to be used by strong learner
    weak_learner  = [gpoint1[0], 
                     gpoint1[1], 
                     gpoint2[0], 
                     gpoint2[1], alpha]
    # Return weak learner and updated weights
    return weak_learner, w


# Function to combine linear weak classifiers and return activation levels and classification of points
def ada_classifier(ada_learner,x_pt, y_pt):
    # Initialise arrays to store activation levels and associated label classifications
    activation_level = []
    classification = []
    # Loop through each data point passed in
    for i in range(len(x_pt)):
        # Initialise point levels to zero for each iteration
        point_level=0
        # Loop through number of weak classifiers
        for j in range(len(ada_learner)):
            # Sum the total classification values for all points in each weak classifier
            point_level += (ada_learner[j][4]) * (classify([x_pt[i],y_pt[i]], ada_learner[j][:2],ada_learner[j][2:4]))
        # Append activation levels for each weak classifier
        activation_level.append(point_level)
        classification.append(np.sign(point_level))
    # Return activation levels and classified labels
    return activation_level, classification


# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


print("Training classifier")
# Load training dataset as numpy array & create points and labels arrays
data_train = np.loadtxt("adaboost-train-20.txt", dtype = float)
points_train, labels_train = data_train[:, :2], data_train[:, 2]

# Load testing dataset as numpy array & create points and labels arrays
data_test = np.loadtxt("adaboost-test-20.txt", dtype = float)
points_test, labels_test = data_test[:, :2], data_test[:, 2]

# Length of training data
N = len(points_train)

# Weights initialised as equal and summing to 1
w = np.full(N, (1 / N))
# define number of iterations
iterations = 8
# initialise string learner array
ada_learner = []

#iterating through the selected amount of classifiers (30)
for i in range(iterations):
    #calacuating the weighted means slop and constant
    m, b = calc_mean_weighted(data_train, w)
    #Finding the midpoints between the projected points
    decision_points = find_midpoints(data_train,m ,b)

    #training the weak classifier
    weak_learner, new_w = train(data_train, decision_points, w)
    #overwrite weights array with new normalized weights
    w = new_w/sum(new_w)
    #storing the weak classifiers
    ada_learner.append(weak_learner)

#Using the adaboost classifier, to classify the training data
act_levels, classifications = ada_classifier(ada_learner, data_train[:,0], data_train[:,1])

#calculating and printing the accuracy
accuracy = np.sum(classifications == labels_train)
percentage = ((accuracy*100)/len(data_train))
print(f"Using {i+1} weak linear classifiers an accuracy of")
print(f"\r{percentage}% was achieved on the final iteration of the training dataset" )

# Run AdaBoost classifier on testing data and graph accuracy vs number of weak learners
testing_accuracy = []
for i in range(len(ada_learner)):
    act_levels_test, classif_test = ada_classifier(ada_learner[:i], data_test[:,0], data_test[:,1])
    num_correct = np.sum(classif_test == labels_test)
    testing_accuracy.append((num_correct*100)/len(labels_test))

# Print info to console relating to final accuracy on Training dataset
print(f"Highest Accuracy on Testing Data: {np.max(testing_accuracy)}% accuracy was acheived")
print(f"using {np.argmax(testing_accuracy)+1} weak learners")

# Create plot for accuracy vs number of linear classifiers
plt.figure(1, [10,10])
plt.title(' Accuracy vs number of weak classifiers on testing data ')

# Create x points that represent the number of linear classifiers
x = list(range(1,len(ada_learner)+1))
plt.plot(x, testing_accuracy)
plt.xlabel("Number of Weak Learners")
plt.ylabel("accuracy")

# Create and store grid points into 2 column matrices x and y to get activation for each point
x, y = np.mgrid[-3:4:0.05,-3:4:0.05]
xgrid = np.reshape(x, (x.size, -1))
ygrid = np.reshape(y, (y.size, -1))

#Use the strong classifier to get activation levels for points on the meshgrid and store in z
z,a=ada_classifier(ada_learner,xgrid, ygrid)
#reshape z for grid
z = np.reshape(z, (x.shape))
#plot contour plot with training data
plt.figure(2, [10,10])
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.title('Contour plot of AdaBoost classifier')
# Plot line where contour that fits activation levels equals zero
plt.contour(x, y, z, levels=[0], colors=["black"])
# Fill regions either side of plotted contour line
plt.contourf(x, y, z, levels=[-1e5, 0, 1e5], colors=["skyblue","lightsalmon"])
plt.scatter(data_test.T[0], data_test.T[1], c=data_test.T[2], cmap='bwr', marker = '1')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid()
plt.axis('equal')

#Plotting a 3D representaion of the of the adaboost decision boundary
plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
plt.title('3D represenation of weak Classifiers')
ax.contour3D(x,y,z,200)
ax.contour(x,y,z,levels=[-1e5,0,1e5],colors=["white","green","red"], linestyles = ['dashed', 'solid', 'dashed'])
ax.scatter3D(data_test.T[0], data_test.T[1], c=data_test.T[2], cmap='bwr')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")
ax.grid()

plt.show()

