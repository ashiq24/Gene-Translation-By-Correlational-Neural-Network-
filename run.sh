#python3 main.py foldertofiles pca_components number_of_copula_data activation last_layer_activation train_op test_op lambda
#train_op = 1 simple training 
#train_op = 2 , train with unique data
#test_op = 1 test left to right generation
#test_op = 2 test right to left generation
#lambda is corrnet hyperparameter, adust between MSE and Correlation
python3 main.py /media/ashiq/Education/Research/DeepSavior/DATA\ Base/gtex6/gtex-adipose-skin/original/Data/ 20 100 selu linear 1 2 30