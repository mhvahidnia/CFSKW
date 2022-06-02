# This code was developed by Mohammad H. Vahidnia in 2022 for POI recommendation
# in LBSN using the CFSKW method. For more information refer to the following
# paper:
# Vahidnia, M.H.
# "Point-of-Interest Recommendation in Location-based Social Networks
# Based on Collaborative Filtering and Spatial Kernel Weighting."
# Geocarto International

# ---------------------------------mthod 1 ----------------------------------------------------------------------------------------------
import pandas as pd 
import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
import math
from sklearn.neighbors import DistanceMetric
from math import radians
from joblib import Parallel, delayed

from google.colab import drive
drive.mount('/content/drive')

#checkins=pd.read_csv('/content/drive/My Drive/Colab Notebooks/dataset_TSMC2014_NYC.csv', encoding= 'unicode_escape', nrows=65114)
checkins=pd.read_csv('/content/drive/My Drive/Colab Notebooks/dataset_TSMC2014_TKY.csv', encoding= 'unicode_escape', nrows=63728)
checkins['UserID'] = pd.factorize(checkins['UserID'])[0] + 1
checkins['VenueID'] = pd.factorize(checkins['VenueID'])[0] + 1

print(checkins.head()) # 
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
print(checkins.shape)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
print(checkins.dtypes)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
print(checkins.describe())
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
print(checkins.index)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

n_users = checkins.UserID.nunique()
print("Number of users: ", n_users)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

n_venues = checkins.VenueID.nunique()
print("Number of venuse: ", n_venues)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

user_venues=checkins.groupby(["UserID","VenueID"])["VenueID"].count()
user_venues=pd.DataFrame(user_venues)

#user_venues.columns = ["UserID", "VenueID","Score"]
user_venues.rename(columns={'UserID': 'UserID', 'VenueID': 'VenueID', 'VenueID': 'Score'}, inplace=True)
print('user_venues:')
print(user_venues)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
smax=max(user_venues['Score']) 
smin=min(user_venues['Score'])
for lab, row in user_venues.iterrows():
    row['Score']=math.ceil((row['Score']-smin)*4/(smax-smin))+1  

print('max score', max(user_venues['Score']))
print('min score', min(user_venues['Score']))
user_venues.to_csv('/content/drive/My Drive/Colab Notebooks/user_venues_frequencies.csv')
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

checkins_dropout = checkins.drop_duplicates(subset='VenueID', keep="first")
checkins_dropout.set_index('VenueID', inplace=True, drop=True)
#checkins_dropout.reset_index()
checkins_dropout.to_csv('/content/drive/My Drive/Colab Notebooks/Unique_Venues.csv')

checkins_dropout['Lat'] = np.radians(checkins_dropout['Lat']) 
checkins_dropout['Long'] = np.radians(checkins_dropout['Long']) 
print('checkins_dropout')
print(checkins_dropout)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
print('checkins_dropout.shape')
print(checkins_dropout.shape)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

dist = DistanceMetric.get_metric('haversine')
dist_mat=dist.pairwise(checkins_dropout[['Lat','Long']].to_numpy())*6373

print(dist_mat)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
print(dist_mat.shape)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

dist_mat=np.round(dist_mat,2)
print(dist_mat)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

print("Max Dist: ", np.max(dist_mat))
print("Max Dist: ", np.min(dist_mat))
h= 8

kernel_type='Gaussian distance-decay-based weighting'

proximity=np.exp(-dist_mat**2/(2*h**2)) 

#proximity=1-(dist_mat-np.min(dist_mat))/(np.max(dist_mat)-np.min(dist_mat)) # a closeness measure based on 1-distance
print(proximity)
print('MAX DISTANCE =', np.max(proximity))
print('MIN DISTANCE =', np.min(proximity))

#dist_mat=pd.DataFrame(dist_mat)
#dist_mat.to_csv('/content/drive/My Drive/Colab Notebooks/dist_mat.csv')

print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

user_venues=pd.read_csv('/content/drive/My Drive/Colab Notebooks/user_venues_frequencies.csv')
print(user_venues.head())

n_users=user_venues.UserID.unique().shape[0] #
n_venues=user_venues.VenueID.unique().shape[0] #

print(str(n_users) + ' users')
print(str(n_venues)+ ' venues')

print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

user_venues_matrix = np.zeros((n_users, n_venues))
for row in user_venues.itertuples():
    user_venues_matrix[row[1]-1, row[2]-1] = row[3]

print('user_venues_matrix')
print(user_venues_matrix)

print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

user_venues_matrix=user_venues_matrix[np.where(np.count_nonzero(user_venues_matrix, axis=1)>15)]

print(user_venues_matrix.shape) 
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

sparsity = float(len(user_venues_matrix.nonzero()[0]))
sparsity /= (user_venues_matrix.shape[0] * user_venues_matrix.shape[1])
sparsity *= 100
print ('Sparsity: {:4.2f}%'.format(sparsity))
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

test = np.zeros(user_venues_matrix.shape)
train = user_venues_matrix.copy()
for user in range(user_venues_matrix.shape[0]):
    test_size =int(len(user_venues_matrix[user, :].nonzero()[0])*20/100) 
    #print(test_size)
    test_ratings = np.random.choice(user_venues_matrix[user, :].nonzero()[0], 
                                    size=test_size, 
                                    replace=False)
    #print(user, test_ratings)
    train[user, test_ratings] = 0.
    test[user, test_ratings] = user_venues_matrix[user, test_ratings]
    
# Test and training are truly disjoint
assert(np.all((train * test) == 0))
print(train, test)


#np.savetxt("/content/drive/My Drive/Colab Notebooks/train2.csv", train, delimiter=",") 
#np.savetxt("/content/drive/My Drive/Colab Notebooks/test2.csv", test, delimiter=",") 
train = np.loadtxt('/content/drive/My Drive/Colab Notebooks/train2.csv', delimiter=',')
test = np.loadtxt('/content/drive/My Drive/Colab Notebooks/test2.csv', delimiter=',')

print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")


user_similarity = 1-pairwise_distances(train, metric='cosine')
item_similarity = 1-pairwise_distances(train.T, metric='cosine')
print('user_similarity ', user_similarity.shape)
print(user_similarity)
print('item_similarity ', item_similarity.shape)
print(item_similarity)
print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff)\
        / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        mean_item_rating = ratings.mean(axis=0)
        ratings_diff = (ratings - mean_item_rating[np.newaxis, :])
        pred = mean_item_rating[np.newaxis, :] + ratings_diff.dot(similarity)\
        / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train, item_similarity, type='item')
user_prediction = predict(train, user_similarity, type='user')
print('user_prediction ', user_prediction.shape)
print(user_prediction)
print('item_prediction ', item_prediction.shape)
print(item_prediction)

spatial_prediction=np.array(train).dot(np.array(proximity))/np.array([np.array(proximity).sum(axis=1)])

alpha=0.8 
beta=1-alpha  

item_spatial_prediction1=alpha*item_prediction+beta*spatial_prediction
user_spatial_prediction1=alpha*user_prediction+beta*spatial_prediction

alpha=0.6 
beta=1-alpha  

item_spatial_prediction2=alpha*item_prediction+beta*spatial_prediction
user_spatial_prediction2=alpha*user_prediction+beta*spatial_prediction

alpha=0.4 
beta=1-alpha  

item_spatial_prediction3=alpha*item_prediction+beta*spatial_prediction
user_spatial_prediction3=alpha*user_prediction+beta*spatial_prediction

alpha=0.2 
beta=1-alpha 

item_spatial_prediction4=alpha*item_prediction+beta*spatial_prediction
user_spatial_prediction4=alpha*user_prediction+beta*spatial_prediction

print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

print('h = ', h)
print('kernel type = ', kernel_type)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

def evl(predection, test):
  prediction_1D = predection[test.nonzero()].flatten() # convert to 1 dimensional array
  #print(user_prediction_1D)
  test_1D = test[test.nonzero()].flatten()
  RMSE_U=sqrt(mean_squared_error(prediction_1D, test_1D))
  MAE_U=mean_absolute_error(prediction_1D, test_1D)
  return RMSE_U, MAE_U

RMSE_U, MAE_U=evl(user_prediction,test)
print("test_RMSE_User_sim = ", RMSE_U)
print("test_MAE_User_sim = ", MAE_U)

RMSE_U, MAE_U=evl(user_spatial_prediction1,test)
print("test_RMSE_User_sim_SPATIAL alpha = 0.2 : ", RMSE_U)
print("test_MAE_User_sim_SPATIAL alpha = 0.2 : ", MAE_U)

RMSE_U, MAE_U=evl(user_spatial_prediction2,test)
print("test_RMSE_User_sim_SPATIAL alpha = 0.4 : ", RMSE_U)
print("test_MAE_User_sim_SPATIAL alpha = 0.4 : ", MAE_U)

RMSE_U, MAE_U=evl(user_spatial_prediction3,test)
print("test_RMSE_User_sim_SPATIAL alpha = 0.6 : ", RMSE_U)
print("test_MAE_User_sim_SPATIAL alpha = 0.6 : ", MAE_U)

RMSE_U, MAE_U=evl(user_spatial_prediction4,test)
print("test_RMSE_User_sim_SPATIAL alpha = 0.8 : ", RMSE_U)
print("test_MAE_User_sim_SPATIAL alpha = 0.8 : ", MAE_U)

RMSE_I, MAE_I=evl(item_prediction,test)
print("test_RMSE_ITEM_sim = ", RMSE_I)
print("test_MAE_ITEM_sim = ", MAE_I)

RMSE_I, MAE_I=evl(item_spatial_prediction1,test)
print("test_RMSE_ITEM_sim_SPATIAL alpha = 0.2 : ", RMSE_I)
print("test_MAE_ITEM_sim_SPATIAL alpha = 0.2 : ", MAE_I)

RMSE_I, MAE_I=evl(item_spatial_prediction2,test)
print("test_RMSE_ITEM_sim_SPATIAL alpha = 0.4 : ", RMSE_I)
print("test_MAE_ITEM_sim_SPATIAL alpha = 0.4 : ", MAE_I)

RMSE_I, MAE_I=evl(item_spatial_prediction3,test)
print("test_RMSE_ITEM_sim_SPATIAL alpha = 0.6 : ", RMSE_I)
print("test_MAE_ITEM_sim_SPATIAL alpha = 0.6 : ", MAE_I)

RMSE_I, MAE_I=evl(item_spatial_prediction4,test)
print("test_RMSE_ITEM_sim_SPATIAL alpha = 0.8 : ", RMSE_I)
print("test_MAE_ITEM_sim_SPATIAL alpha = 0.8 : ", MAE_I)


def quality(prediction,test):
  overall_precision_at_5=[]
  overall_recall_at_5=[]
  overall_precision_at_10=[]
  overall_recall_at_10=[]
  overall_precision_at_15=[]
  overall_recall_at_15=[]

  i=0
  for row in prediction:
        pred_actual=[]  
        relevant_1=set()  
        relevant_2=set()  
        relevant_3=set()  
        recomm_1=set()  
        recomm_2=set() 
        recomm_3=set()  
        pred_actual.append(test[i])
        pred_actual.append(row)
        i+=1
        pred_actual = np.array(pred_actual)
        pred_actual = np.flip(pred_actual[:, pred_actual[1].argsort()]) 

        pred_actual = pred_actual.transpose()
        k=0
        for item in pred_actual:
            if item[1]==0: 
                #print(k)
                pred_actual=np.delete(pred_actual, k, 0)
            else:
                if item[1]>=2 and len(relevant_1)<5:
                    relevant_1.add(k)
                if item[1]>=2 and len(relevant_2)<10:
                    relevant_2.add(k)
                if item[1]>=2 and len(relevant_3)<15:
                    relevant_3.add(k)
                if len(recomm_1)<5:
                    recomm_1.add(k)
                if len(recomm_2)<10:
                    recomm_2.add(k)
                if len(recomm_3)<15:
                    recomm_3.add(k)                  
                k+=1
        pred_actual = pred_actual.transpose()
        intersect1=relevant_1 & recomm_1
        intersect2=relevant_2 & recomm_2
        intersect3=relevant_3 & recomm_3

        if len(relevant_1)!=0:
              precesion_at_5=len(intersect1)/5*100
              recall_at_5=len(intersect1)/len(relevant_1)*100
        else:
              precesion_at_5=0
              recall_at_5=0
        if len(relevant_2)!=0:
              precesion_at_10=len(intersect2)/10*100
              recall_at_10=len(intersect2)/len(relevant_2)*100
        else:
              precesion_at_10=0
              recall_at_10=0
        if len(relevant_3)!=0:
              precesion_at_15=len(intersect3)/15*100
              recall_at_15=len(intersect3)/len(relevant_3)*100
        else:
              precesion_at_15=0
              recall_at_15=0

        overall_recall_at_5.append(recall_at_5)
        overall_precision_at_5.append(precesion_at_5)
        overall_recall_at_10.append(recall_at_10)
        overall_precision_at_10.append(precesion_at_10)
        overall_recall_at_15.append(recall_at_15)
        overall_precision_at_15.append(precesion_at_15)

  overall_recall_at_5=sum(overall_recall_at_5) / len(overall_recall_at_5)
  overall_precision_at_5=sum(overall_precision_at_5) / len(overall_precision_at_5)
  overall_recall_at_10=sum(overall_recall_at_10) / len(overall_recall_at_10)
  overall_precision_at_10=sum(overall_precision_at_10) / len(overall_precision_at_10)
  overall_recall_at_15=sum(overall_recall_at_15) / len(overall_recall_at_15)
  overall_precision_at_15=sum(overall_precision_at_15) / len(overall_precision_at_15)

  return overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15

print("-.-.-.-.-.-.-.-.-.-.-.-.USER BASED.-.-.-.-.-.-.-.-.-.-.-.")

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(user_prediction,test)

print("+++++++++++USER BASED COLLABORATIVE FILTERING++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(user_spatial_prediction1,test)

print("+++++++++++USER BASED COLLABORATIVE FILTERING & GEOGRAPHICAL KERNEL WEIGHTING, alpha = 0.2 ++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(user_spatial_prediction2,test)

print("+++++++++++USER BASED COLLABORATIVE FILTERING & GEOGRAPHICAL KERNEL WEIGHTING, alpha = 0.4 ++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(user_spatial_prediction3,test)

print("+++++++++++USER BASED COLLABORATIVE FILTERING & GEOGRAPHICAL KERNEL WEIGHTING, alpha = 0.6 ++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(user_spatial_prediction4,test)

print("+++++++++++USER BASED COLLABORATIVE FILTERING & GEOGRAPHICAL KERNEL WEIGHTING, alpha = 0.8 ++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

print("-.-.-.-.-.-.-.-.-.-.-.-ITEM BASED.-.-.-.-.-.-.-.-.-.-.")

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(item_prediction,test)

print("+++++++++++USER BASED COLLABORATIVE FILTERING++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(item_spatial_prediction1,test)

print("+++++++++++ ITEM BASED COLLABORATIVE FILTERING & GEOGRAPHICAL KERNEL WEIGHTING, alpha = 0.2 ++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(item_spatial_prediction2,test)

print("+++++++++++ ITEM BASED COLLABORATIVE FILTERING & GEOGRAPHICAL KERNEL WEIGHTING, alpha = 0.4 ++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(item_spatial_prediction3,test)

print("+++++++++++ ITEM BASED COLLABORATIVE FILTERING & GEOGRAPHICAL KERNEL WEIGHTING, alpha = 0.6 ++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

overall_recall_at_5, overall_precision_at_5, overall_recall_at_10, overall_precision_at_10, overall_recall_at_15, overall_precision_at_15 \
= quality(item_spatial_prediction4,test)

print("+++++++++++ ITEM BASED COLLABORATIVE FILTERING & GEOGRAPHICAL KERNEL WEIGHTING, alpha = 0.8 ++++++++++++++")
print("overall_recall_at_5 = ", overall_recall_at_5)
print("overall_precision_at_5 = ", overall_precision_at_5)
print("overall_recall_at_10 = ", overall_recall_at_10)
print("overall_precision_at_10 = ", overall_precision_at_10)
print("overall_recall_at_15 = ", overall_recall_at_15)
print("overall_precision_at_15 = ", overall_precision_at_15)

