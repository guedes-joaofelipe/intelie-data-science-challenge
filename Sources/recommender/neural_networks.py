# __author__ = 'Joao Felipe Guedes <guedes.joaofelipe@poli.ufrj.br>'

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import optimizers
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd 
import numpy as np

class AE:
    def __init__(self, train_set=None, test_set=None, random_seed=None, num_neurons=10, 
                 epochs = 2000, learning_rate=0.001, loss_metric='mse', training_metrics=['mae'],
                 patience=10, kernel_initializer='random_uniform'):
        
        self.recommender_name = 'AE'  
        self.train_set = train_set
        self.test_set = test_set
        self.learning_rate = learning_rate        
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.loss_metric = loss_metric  
        self.training_metrics = training_metrics
        self.patience = patience
        self.kernel_initializer = kernel_initializer

        if random_seed is not None:
            np.random.seed(random_seed)

        # internal vars      
        self.matrix = None
        self.matrix_normalized = None
        self.matrix_test = None
        self.matrix_normalized_test = None
        self.prediction_matrix = None
        self.prediction_matrix_test = None        
        
    def fit(self):
        """
        This method performs an Autoencoder to reconstruct the utility matrix over the training data.
        """
        self.init_model()
        
        self.Adam = optimizers.Adam(lr = self.learning_rate)

        self.autoencoder.compile(optimizer = self.Adam, loss = self.loss_metric, metrics=self.training_metrics)

        if self.test_set is not None:            
            self.fit_history = self.autoencoder.fit(
                                        self.matrix_normalized, 
                                        self.matrix_normalized, 
                                        epochs = self.epochs,
                                        verbose = 0,
                                        shuffle = True,
                                        validation_data = (self.matrix_normalized_test, self.matrix_normalized_test), 
                                        callbacks = [self.earlyStopping])            
        else:
            self.fit_history = self.autoencoder.fit(
                                        self.matrix_normalized, 
                                        self.matrix_normalized, 
                                        epochs = self.epochs,
                                        verbose = 0,
                                        shuffle = True,                                        
                                        callbacks = [self.earlyStopping])

        # Predicting output for normalized input matrix
        output_normalized = self.autoencoder.predict(self.matrix_normalized);

        # Denormalizing output
        output_denormalized = self.norm_scaler.inverse_transform(output_normalized);

        self.prediction_matrix = output_denormalized

    def init_model(self):
        """
        Method to treat and initialize the model
        """       
        # Fitting matrix scaler for the range [-1, 1]        
        self.norm_scaler = MinMaxScaler(feature_range = (-1,1))
        self.create_matrix()
        
        #Setting autoencoder
        self.input_dim = self.matrix.as_matrix().shape[1] 

        self.autoencoder = Sequential([
                                  Dense(self.num_neurons, activation = 'selu', kernel_initializer = self.kernel_initializer, input_dim = self.input_dim),
                                  Dense(self.input_dim, activation = 'tanh', kernel_initializer = self.kernel_initializer)
                                 ])

        self.earlyStopping = EarlyStopping(monitor = 'loss', patience = self.patience, mode = 'auto')
        
    def create_matrix(self):
        """ Creates a matrix dataframe having users as rows 
            an items as columns according to combined training and test set (if applicable)"""
        
        if self.test_set is None:
            self.matrix = self.train_set.pivot(index='user_id', columns='item_id', values='feedback').fillna(0)
        else:
            # Combining both training and test set
            df_temp = self.test_set.copy()
            df_temp['feedback'] = 0
            df_temp['timestamp'] = 0
            self.matrix = self.train_set.append(df_temp).reset_index(drop = True).pivot(index='user_id', columns='item_id', values='feedback').fillna(0)
            
            df_temp = self.train_set.copy()
#             df_temp['feedback'] = 0
#             df_temp['timestamp'] = 0
            self.matrix_test = df_temp.append(self.test_set).reset_index(drop = True).pivot(index='user_id', columns='item_id', values='feedback').fillna(0)
                
        self.norm_scaler.fit(self.matrix)
        self.matrix_normalized = self.norm_scaler.transform(self.matrix.as_matrix())                
                
        if self.test_set is not None:
            self.matrix_normalized_test = self.norm_scaler.transform(self.matrix_test.as_matrix())                
            
            
    def get_predicted_matrix(self, test_matrix=False):
        # Returns the predicted matrix without normalization in a DataFrame type
        # If test_matrix is set to True, prediction will be applied to normalized test matrix
        if not test_matrix:
            predicted_matrix = self.norm_scaler.inverse_transform(self.autoencoder.predict(self.matrix_normalized))
            return pd.DataFrame(predicted_matrix, columns=self.matrix.columns, index=self.matrix.index)
        
        predicted_matrix = self.norm_scaler.inverse_transform(self.autoencoder.predict(self.matrix_normalized_test))
        return pd.DataFrame(predicted_matrix, columns=self.matrix.columns, index=self.matrix.index)
    
    
    def get_loss(self, test_matrix=False, masked=True):
        # Returns the MAE and RMSE after fitting the model. 
        # If masked is set to true, null ratings in the input matrix will not be taken into account
        subject_matrix = self.matrix_test
        if not test_matrix:            
            subject_matrix = self.matrix
        
        if (masked):
            zero_elements_index = np.where(subject_matrix.as_matrix().flatten() != 0)
            y_true = subject_matrix.as_matrix().flatten()[zero_elements_index]
            y_pred = self.get_predicted_matrix(test_matrix).as_matrix().flatten()[zero_elements_index]
        else:
            y_true = subject_matrix.as_matrix().flatten()
            y_pred = self.get_predicted_matrix(test_matrix).as_matrix().flatten()
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred)        )
        return {'mae': mae, 'rmse': rmse}
            
    def get_user_prediction(self, user_id, item_id):
        # returns the rating prediction of a pair user-item
        if not self.check_user_item_id(user_id, item_id):
            return None
        else:
            return self.get_predicted_matrix().iloc[user_id - 1][item_id]

    def get_user_top_n(self, user_id, n):
        # Get the top n not consumed items for a target user id 
        if not self.check_user_item_id(user_id=user_id):
            return None
        else:                
            zero_elements_index = np.where(self.matrix.iloc[user_id-1] == 0)
            zero_ratings_item_id = self.matrix.columns[zero_elements_index]
        return np.array(self.get_predicted_matrix().iloc[user_id-1][zero_ratings_item_id].argsort()[-n:][::-1].index)

    def check_user_item_id(self, user_id = None, item_id = None):
        # CHecks if user/item id is available in the prediction matrix
        if user_id is not None and user_id not in self.get_predicted_matrix().index:
            print ("User ID {} is not available".format(user_id))
            return False
        elif item_id is not None and item_id not in self.get_predicted_matrix().columns:
            print ("Item ID {} is not available".format(item_id))
            return False
        return True
    
    def save(self, path):
        # Saves the model in specified path
        return self.autoencoder.save(path)
