from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import Gmodel as gmodel

lgbm_c = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0, 
            learning_rate=0.5, max_depth=7, min_child_samples=20, 
            min_child_weight=0.001, min_split_gain=0.0, n_estimators=100, 
            n_jobs=-1, num_leaves=500, objective='binary', random_state=None, 
            reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0, 
            subsample_for_bin=200000, subsample_freq=0)


def model_prediction(algorithm,training_x,testing_x,  training_y,testing_y,cf,threshold_plot):  
    #model  
    algorithm.fit(training_x,training_y)  
    predictions   = algorithm.predict(testing_x)  
    probabilities = algorithm.predict_proba(testing_x)  
    #coeffs  
    if   cf == "coefficients" :  
        coefficients  = pd.DataFrame(algorithm.coef_.ravel())  
    elif cf == "features" :  
        coefficients  = pd.DataFrame(algorithm.feature_importances_)  
    coef_sumry    = coefficients 
    coef_sumry.columns = ["coefficients","features"]  
    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)  
    print (algorithm)  
    print ("\n Classification report : \n",classification_report(testing_y,predictions))  
    print ("Accuracy   Score : ",accuracy_score(testing_y,predictions))  
    #confusion matrix  
    conf_matrix = confusion_matrix(testing_y,predictions)  
    #roc_auc_score  
    model_roc_auc = roc_auc_score(testing_y,predictions)   
    print ("Area under curve : ",model_roc_auc,"\n")  
    fpr,tpr,thresholds = roc_curve(testing_y,probabilities[:,1])  
    #plot confusion matrix  
    trace1 = go.Heatmap(z = conf_matrix ,  
                        x = ["Negative","Positive"],  
                        y = ["Negative","Positive"],  
                        showscale  = False,colorscale = "Picnic",  
                        name = "matrix")  
    #plot roc curve  
    trace2 = go.Scatter(x = fpr,y = tpr,  
                        name = "Roc : " + str(model_roc_auc),  
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2))  
    trace3 = go.Scatter(x = [0,1],y=[0,1],  
                        line = dict(color = ('rgb(205, 12, 24)'),width = 2,  
                        dash = 'dot'))  
    #plot coeffs  
    trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],  
                    name = "coefficients",  
                    marker = dict(color = coef_sumry["coefficients"],  
                                    colorscale = "Picnic",  
                                    line = dict(width = .6,color = "black")))  
    #subplots  
    fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],  
                            subplot_titles=('Confusion Matrix',  
                                            'Receiver operating characteristic',  
                                            'Feature Importances'))  
    fig.append_trace(trace1,1,1)  
    fig.append_trace(trace2,1,2)  
    fig.append_trace(trace3,1,2)  
    fig.append_trace(trace4,2,1)  
    fig['layout'].update(showlegend=False, title="Model performance" ,  
                            autosize = False,height = 900,width = 800,  
                            plot_bgcolor = 'rgba(240,240,240, 0.95)',  
                            paper_bgcolor = 'rgba(240,240,240, 0.95)',  
                            margin = dict(b = 195))  
    fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))  
    fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))  
    fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),  
                                        tickangle = 90))  
    display(fig.show())  
    if threshold_plot == True :   
        visualizer = DiscriminationThreshold(algorithm)  
        visualizer.fit(training_x,training_y)  
        visualizer.poof()



clf=XGBClassifier()
model_prediction(lgbm_c , model_assistant.X_train , model_assistant.X_test , model_assistant.y_train
                                ,model_assistant.y_test ,'features',0.5                              
                                )

model_prediction(clf,X_train,X_test,Y_train,Y_test,'features',0.5)