from sklearn import svm


x = [[2, 0], [1, 1], [2, 3]]
y = [0, 0, 1]  ######0 means "-1"
clf = svm.SVC(kernel = 'linear')####classifier####
clf.fit(x, y)

print("detial for classifier:\n", clf)

# get support vectors
print("clf.support_vectors_(which 'point' support the vector (edge lines)?):\n", clf.support_vectors_)

# get indices of support vectors
print('clf.support_,(the "point" start with "0"):\n', clf.support_)

# get number of support vectors for each class
print('clf.n_support_(How many "point" does support the vector (edge lines)in different side of servial D?)\n', clf.n_support_)

# predict the "point"
# print("a =  ")
# a = input()
# print("b =  ")
# b = input()

print(" predict y of a new 'point()' :\n", clf.predict([2,0]))