Program is can be run through one of 2 paths - one for getting the value of most optimal k and other for normal output for a test set.

The path to test and training set should be mentioned on line 261(optimal k value) or 263(normal running)by the user

and if the value of k needs to be changed then please edit it on line 252

currently implemented a weighted sum approach where the k highest similarities are multiplied with the classificaiton associated with associated review and added together with their signs

if the sum is positive, '+1' is added as the classificaiton of the test input, else '-1'

