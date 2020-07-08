import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
#image = Image.open('/Users/rajath/Projects/Diabetes/triangle.png')
#st.image(image, width = 100)

#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    #st.markdown("<h1 style='text-align: left; color: black;'>Blaze Model Selector</h1>", unsafe_allow_html=True)
    st.title("Blaze Model Selector")
    st.sidebar.title("Hi There!")
    #st.sidebar.markdown("Hope you enjoy 😃")
    st.header("Choose the Best Classification Model for your Dataset!")
    st.subheader("Upload your Classifier Dataset with the format X1, X2, X3.... Xn , Outcome ")
    st.markdown("Where X are the features & Outcome is the Output Vector Y Containing values 0 or 1 ")
    st.markdown("Please make sure all the values are Integer/Float & there are NO missing values.")
    st.markdown("Sample dataset is provided below ✌️")

    uploaded_file = st.file_uploader("Upload the CSV", type="csv")

#LOADING FILE. * CHANGE INTO USER INPUT
    @st.cache(persist = True)
    def sample_load_data():
        sample_data = pd.read_csv('sampleDiabetes.csv')
        return sample_data

    sample_df = sample_load_data()

    def load_data():
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except:
            print('Upload a Proper File ')
    if st.checkbox("Sample dataset", False):
        st.subheader("Sample Diabetes Dataset")
        st.write(sample_df)

    st.text('If u see a red error box below. Please Upload the CSV file with the proper format')



#ASSIGNING FEATURES X AND OUTPUT VECTOR Y
#SPLITTING DATASET
    @st.cache(persist=True)
    def split(df):
        y=df.Outcome.astype(float)
        x=df.drop(columns=['Outcome']).astype(float)
        x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
        return x_train, x_test,y_train,y_test
        st.write(df)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
        imputer.fit(x) # 1:3 . 3 excluded.
        x = imputer.transform(x)


#PLOTTING MATRICES
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix :")
            plot_confusion_matrix(model, x_test,y_test,display_labels = class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC curve :")
            plot_roc_curve(model, x_test,y_test)
            st.pyplot()

        if 'Precision Recall Curve' in metrics_list:
            st.subheader("Precision Recall Curve:")
            plot_precision_recall_curve(model, x_test,y_test)
            st.pyplot()


#CALLING FUNCTIONS & CHOOSING CLASSIFIER

    df = load_data()
    x_train,x_test,y_train,y_test = split(df)
    class_names = ['Positive','Negative']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine","LogisticRegression","RandomForestClassifier","GaussianNB","KNeighborsClassifier","SGD_Classifier"))


#SUPPORT VECTOR MACHINE
    if classifier == 'Support Vector Machine':
        st.sidebar.subheader("Select Hyperparameters")
        C = st.sidebar.number_input("C = Regularisation Parmeter (Lower the better)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel",("rbf","linear"), key = 'kernel')
        gamma = st.sidebar.radio("Gamma(Kernel Co-eff)",("scale","auto"),key='gamma')

        metrics = st.sidebar.multiselect("Choose the metrics : ",('Confusion Matrix','ROC Curve','Precision Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("SVM Results : ")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ", precision_score(y_test, y_pred, labels= class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels= class_names).round(2))
            plot_metrics(metrics)

#LogisticRegression

    if classifier == 'LogisticRegression':
        st.sidebar.subheader("Select Hyperparameters")
        C = st.sidebar.number_input("C = regularisation Parameter (Lower the better)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100,500,key='max_iter')
        metrics = st.sidebar.multiselect("Choose the metrics : ",('Confusion Matrix','ROC Curve','Precision Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("LogisticRegression Results : ")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ", precision_score(y_test, y_pred, labels= class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels= class_names).round(2))
            plot_metrics(metrics)

#RANDOM FOREST CLASSIFIER

    if classifier == 'RandomForestClassifier':
        st.sidebar.subheader("Select Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in forest",100,5000,step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,step=1,key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap Samples whe building Trees : ",('True','False'),key='bootstrap')
        metrics = st.sidebar.multiselect("Choose the metrics : ",('Confusion Matrix','ROC Curve','Precision Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("RandomForestClassifier Results : ")
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ", precision_score(y_test, y_pred, labels= class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels= class_names).round(2))
            plot_metrics(metrics)

#GaussianNB

    if classifier == 'GaussianNB':

        metrics = st.sidebar.multiselect("Choose the metrics : ",('Confusion Matrix','ROC Curve','Precision Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("GaussianNB Results : ")
            model = GaussianNB()
            model.fit(x_train, y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ", precision_score(y_test, y_pred, labels= class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels= class_names).round(2))
            plot_metrics(metrics)

#KNeighborsClassifier
    if classifier == 'KNeighborsClassifier':
        st.sidebar.subheader("Select Hyperparameters")
        n_neighbors = st.sidebar.slider("The number of neighbors to consiider : ",1,10,step=1,key='n_neighbors')
        algorithm = st.sidebar.selectbox("Algorithm",("auto","ball_tree","kd_tree","brute"),key='algorithm')
        metrics = st.sidebar.multiselect("Choose the metrics : ",('Confusion Matrix','ROC Curve','Precision Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("K Nearest Results : ")
            model = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=algorithm)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ", precision_score(y_test, y_pred, labels= class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels= class_names).round(2))
            plot_metrics(metrics)

#Stochastic Gradient Descent
    if classifier == 'SGD_Classifier':
        st.sidebar.subheader("Default Hyperparameters Set")

        #alpha = st.sidebar.selectbox("Regularization Parameter: ",("0.001","0.01","0.1","0.5"),key='alpha')
        #max_iter = st.sidebar.number_input("Maximum number of iterations", 1000,5000,key='max_iter')
        metrics = st.sidebar.multiselect("Choose the metrics : ",('Confusion Matrix','ROC Curve','Precision Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("SGDClassifier Results : ")
            model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0,
            epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
            class_weight=None, warm_start=False, average=False)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy : ", accuracy.round(2))
            st.write("Precision : ", precision_score(y_test, y_pred, labels= class_names).round(2))
            st.write("Recall : ", recall_score(y_test, y_pred, labels= class_names).round(2))
            plot_metrics(metrics)

#RAW DATA FUNCTION




#local_css("styles.css")

if __name__ == "__main__" :
    main()
